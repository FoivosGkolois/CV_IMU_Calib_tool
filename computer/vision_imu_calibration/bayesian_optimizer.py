from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from .imu_filter import KneeAngleFilterParameters

BO_EVALUATION_BUDGET = 30
BO_INITIAL_DESIGN_SIZE = 8
BO_CANDIDATE_POOL_SIZE = 512
BO_RANDOM_SEED = 20260929

_PARAMETER_NAMES = (
    "alpha_stable",
    "alpha_dynamic",
    "max_accel_correction_deg",
)
_PARAMETER_BOUNDS = np.asarray(
    [
        [0.80, 0.95],
        [0.95, 0.99],
        [5.0, 15.0],
    ],
    dtype=float,
)
_GP_LENGTH_SCALE = np.asarray([0.30, 0.30, 0.35], dtype=float)
_GP_NUGGET = 1e-6
_EI_XI_STANDARDIZED = 0.01


@dataclass(frozen=True)
class SearchObservation:
    evaluation_index: int
    parameters: dict[str, float]
    objective_rmse_deg: float
    feasible: bool
    constraint_reasons: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BayesianOptimizationResult:
    parameters: KneeAngleFilterParameters
    metrics: Any
    observations: list[SearchObservation]
    evaluation_budget: int
    random_seed: int
    initial_design_size: int
    candidate_pool_size: int
    gp_uncertainty: dict[str, float]

    def trace_payload(self) -> list[dict[str, Any]]:
        return [observation.to_dict() for observation in self.observations]


Evaluator = Callable[[KneeAngleFilterParameters], Any]
Feasibility = Callable[[Any], tuple[bool, Sequence[str]]]


def parameter_bounds_payload() -> dict[str, list[float]]:
    return {
        name: [float(bounds[0]), float(bounds[1])]
        for name, bounds in zip(_PARAMETER_NAMES, _PARAMETER_BOUNDS)
    }


def optimize_filter_parameters(
    base_parameters: KneeAngleFilterParameters,
    evaluator: Evaluator,
    feasibility: Feasibility,
    *,
    evaluation_budget: int = BO_EVALUATION_BUDGET,
    seed: int = BO_RANDOM_SEED,
) -> BayesianOptimizationResult:
    """Deterministic Gaussian-process Bayesian optimization for three filter parameters.

    The physical/static quantities remain fixed. Only alpha_stable, alpha_dynamic,
    and max_accel_correction_deg are searched. The objective is supplied by the
    caller and is expected to be synchronized training-cycle RMSE.
    """
    budget = max(BO_INITIAL_DESIGN_SIZE, int(evaluation_budget))
    rng = np.random.default_rng(int(seed))
    observations: list[SearchObservation] = []
    evaluated_vectors: list[np.ndarray] = []
    evaluated_metrics: list[Any] = []

    base_vector = _parameters_to_unit_vector(base_parameters)
    initial_vectors = [base_vector]
    initial_vectors.extend(
        _latin_hypercube(
            max(0, min(BO_INITIAL_DESIGN_SIZE, budget) - 1),
            dimensions=len(_PARAMETER_NAMES),
            rng=rng,
        )
    )

    for vector in _deduplicate_vectors(initial_vectors):
        if len(observations) >= budget:
            break
        _evaluate_vector(
            vector,
            base_parameters,
            evaluator,
            feasibility,
            observations,
            evaluated_vectors,
            evaluated_metrics,
        )

    while len(observations) < budget:
        x_train = np.vstack(evaluated_vectors)
        y_train = _surrogate_targets(observations)
        model = _GaussianProcess.fit(x_train, y_train)
        pool = _candidate_pool(rng, x_train, observations)
        if pool.size == 0:
            break
        means, stds = model.predict(pool)
        feasible_objectives = [
            observation.objective_rmse_deg
            for observation in observations
            if observation.feasible and math.isfinite(observation.objective_rmse_deg)
        ]
        if feasible_objectives:
            best = min(feasible_objectives)
            best_standardized = (best - model.y_mean) / model.y_scale
            acquisition = _expected_improvement(
                means,
                stds,
                best_standardized,
                xi=_EI_XI_STANDARDIZED,
            )
        else:
            acquisition = stds
        vector = pool[int(np.argmax(acquisition))]
        _evaluate_vector(
            vector,
            base_parameters,
            evaluator,
            feasibility,
            observations,
            evaluated_vectors,
            evaluated_metrics,
        )

    feasible_indices = [
        index
        for index, observation in enumerate(observations)
        if observation.feasible and math.isfinite(observation.objective_rmse_deg)
    ]
    if not feasible_indices:
        raise RuntimeError("Bayesian optimization produced no feasible filter candidate")
    best_index = min(
        feasible_indices,
        key=lambda index: observations[index].objective_rmse_deg,
    )
    best_parameters = _unit_vector_to_parameters(
        evaluated_vectors[best_index], base_parameters
    )
    best_metrics = evaluated_metrics[best_index]

    x_train = np.vstack(evaluated_vectors)
    final_model = _GaussianProcess.fit(x_train, _surrogate_targets(observations))
    uncertainty_pool = _candidate_pool(rng, x_train, observations, include_local=False)
    best_mean, best_std = final_model.predict(np.asarray([evaluated_vectors[best_index]]))
    if uncertainty_pool.size:
        _pool_mean, pool_std = final_model.predict(uncertainty_pool)
        pool_median_std = float(np.median(pool_std) * final_model.y_scale)
        pool_max_std = float(np.max(pool_std) * final_model.y_scale)
    else:
        pool_median_std = 0.0
        pool_max_std = 0.0
    gp_uncertainty = {
        "recommended_posterior_std_rmse_deg": float(best_std[0] * final_model.y_scale),
        "candidate_pool_median_std_rmse_deg": pool_median_std,
        "candidate_pool_max_std_rmse_deg": pool_max_std,
    }
    return BayesianOptimizationResult(
        parameters=best_parameters,
        metrics=best_metrics,
        observations=observations,
        evaluation_budget=len(observations),
        random_seed=int(seed),
        initial_design_size=min(BO_INITIAL_DESIGN_SIZE, len(observations)),
        candidate_pool_size=BO_CANDIDATE_POOL_SIZE,
        gp_uncertainty=gp_uncertainty,
    )


def sample_random_parameters(
    base_parameters: KneeAngleFilterParameters,
    count: int,
    *,
    seed: int,
) -> list[KneeAngleFilterParameters]:
    target = max(1, int(count))
    rng = np.random.default_rng(int(seed))
    vectors: list[np.ndarray] = [_parameters_to_unit_vector(base_parameters)]
    while len(vectors) < target:
        vectors.append(rng.random(len(_PARAMETER_NAMES)))
    vectors = _deduplicate_vectors(vectors)
    while len(vectors) < target:
        vectors.extend(_deduplicate_vectors([rng.random(len(_PARAMETER_NAMES))]))
        vectors = _deduplicate_vectors(vectors)
    return [
        _unit_vector_to_parameters(vector, base_parameters)
        for vector in vectors[:target]
    ]


def _evaluate_vector(
    vector: np.ndarray,
    base_parameters: KneeAngleFilterParameters,
    evaluator: Evaluator,
    feasibility: Feasibility,
    observations: list[SearchObservation],
    evaluated_vectors: list[np.ndarray],
    evaluated_metrics: list[Any],
) -> None:
    vector = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    parameters = _unit_vector_to_parameters(vector, base_parameters)
    metrics = evaluator(parameters)
    feasible, reasons = feasibility(metrics)
    objective = float(getattr(metrics, "rmse_deg", math.inf))
    if not math.isfinite(objective):
        feasible = False
        reasons = [*list(reasons), "non_finite_rmse"]
    observation = SearchObservation(
        evaluation_index=len(observations) + 1,
        parameters={name: float(getattr(parameters, name)) for name in _PARAMETER_NAMES},
        objective_rmse_deg=objective,
        feasible=bool(feasible),
        constraint_reasons=list(dict.fromkeys(str(reason) for reason in reasons)),
        metrics=_metrics_payload(metrics),
    )
    observations.append(observation)
    evaluated_vectors.append(vector.copy())
    evaluated_metrics.append(metrics)


def _metrics_payload(metrics: Any) -> dict[str, Any]:
    if hasattr(metrics, "to_dict"):
        return dict(metrics.to_dict())
    if hasattr(metrics, "__dict__"):
        return dict(metrics.__dict__)
    return {"value": metrics}


def _surrogate_targets(observations: Sequence[SearchObservation]) -> np.ndarray:
    feasible = [
        observation.objective_rmse_deg
        for observation in observations
        if observation.feasible and math.isfinite(observation.objective_rmse_deg)
    ]
    finite_all = [
        observation.objective_rmse_deg
        for observation in observations
        if math.isfinite(observation.objective_rmse_deg)
    ]
    if feasible:
        worst = max(feasible)
        spread = max(1.0, float(np.std(feasible)))
        penalty = worst + 3.0 * spread
    elif finite_all:
        penalty = max(finite_all) + max(5.0, float(np.std(finite_all)) * 3.0)
    else:
        penalty = 1_000.0
    return np.asarray(
        [
            observation.objective_rmse_deg
            if observation.feasible and math.isfinite(observation.objective_rmse_deg)
            else penalty
            for observation in observations
        ],
        dtype=float,
    )


def _parameters_to_unit_vector(parameters: KneeAngleFilterParameters) -> np.ndarray:
    raw = np.asarray([float(getattr(parameters, name)) for name in _PARAMETER_NAMES])
    lows = _PARAMETER_BOUNDS[:, 0]
    highs = _PARAMETER_BOUNDS[:, 1]
    return np.clip((raw - lows) / (highs - lows), 0.0, 1.0)


def _unit_vector_to_parameters(
    vector: np.ndarray,
    base_parameters: KneeAngleFilterParameters,
) -> KneeAngleFilterParameters:
    vector = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
    lows = _PARAMETER_BOUNDS[:, 0]
    highs = _PARAMETER_BOUNDS[:, 1]
    values = lows + vector * (highs - lows)
    alpha_stable = float(values[0])
    alpha_dynamic = max(alpha_stable, float(values[1]))
    candidate = replace(
        base_parameters,
        alpha_stable=alpha_stable,
        alpha_dynamic=alpha_dynamic,
        max_accel_correction_deg=float(values[2]),
    )
    candidate.validate()
    return candidate


def _latin_hypercube(count: int, *, dimensions: int, rng: np.random.Generator) -> list[np.ndarray]:
    if count <= 0:
        return []
    points = np.zeros((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        permutation = rng.permutation(count)
        points[:, dimension] = (permutation + rng.random(count)) / float(count)
    return [row.copy() for row in points]


def _candidate_pool(
    rng: np.random.Generator,
    evaluated: np.ndarray,
    observations: Sequence[SearchObservation],
    *,
    include_local: bool = True,
) -> np.ndarray:
    global_count = BO_CANDIDATE_POOL_SIZE
    pool = rng.random((global_count, len(_PARAMETER_NAMES)))
    if include_local:
        feasible = [
            (index, observation.objective_rmse_deg)
            for index, observation in enumerate(observations)
            if observation.feasible and math.isfinite(observation.objective_rmse_deg)
        ]
        if feasible:
            best_index = min(feasible, key=lambda item: item[1])[0]
            best = evaluated[best_index]
            local = np.clip(
                best + rng.normal(0.0, 0.08, size=(128, len(_PARAMETER_NAMES))),
                0.0,
                1.0,
            )
            pool = np.vstack([pool, local])
    if evaluated.size:
        distances = np.linalg.norm(pool[:, None, :] - evaluated[None, :, :], axis=2)
        pool = pool[np.all(distances >= 1e-5, axis=1)]
    if pool.size == 0:
        return np.empty((0, len(_PARAMETER_NAMES)), dtype=float)
    rounded = np.round(pool, decimals=10)
    _unique, first_indices = np.unique(rounded, axis=0, return_index=True)
    first_indices.sort()
    return pool[first_indices]

def _deduplicate_vectors(vectors: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for raw in vectors:
        vector = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        if any(np.linalg.norm(vector - previous) < 1e-8 for previous in unique):
            continue
        unique.append(vector.copy())
    return unique


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best: float,
    *,
    xi: float,
) -> np.ndarray:
    improvement = best - mean - float(xi)
    safe_std = np.maximum(std, 1e-12)
    z = improvement / safe_std
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    expected = improvement * cdf + safe_std * pdf
    expected[std <= 1e-12] = 0.0
    return expected


@dataclass(frozen=True)
class _GaussianProcess:
    x_train: np.ndarray
    alpha: np.ndarray
    chol: np.ndarray
    y_mean: float
    y_scale: float

    @classmethod
    def fit(cls, x_train: np.ndarray, y_train: np.ndarray) -> "_GaussianProcess":
        x = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        y_mean = float(np.mean(y))
        y_scale = float(np.std(y))
        if not math.isfinite(y_scale) or y_scale < 1e-9:
            y_scale = 1.0
        standardized = (y - y_mean) / y_scale
        kernel = _rbf_kernel(x, x)
        nugget = _GP_NUGGET
        for _attempt in range(6):
            try:
                chol = np.linalg.cholesky(kernel + np.eye(len(x)) * nugget)
                alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, standardized))
                return cls(
                    x_train=x,
                    alpha=alpha,
                    chol=chol,
                    y_mean=y_mean,
                    y_scale=y_scale,
                )
            except np.linalg.LinAlgError:
                nugget *= 10.0
        raise RuntimeError("Gaussian-process covariance matrix could not be stabilized")

    def predict(self, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(x_query, dtype=float)
        cross = _rbf_kernel(self.x_train, query)
        mean = cross.T @ self.alpha
        solve = np.linalg.solve(self.chol, cross)
        variance = np.maximum(1e-12, 1.0 - np.sum(solve * solve, axis=0))
        return mean, np.sqrt(variance)


def _rbf_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_scaled = np.asarray(left, dtype=float) / _GP_LENGTH_SCALE
    right_scaled = np.asarray(right, dtype=float) / _GP_LENGTH_SCALE
    delta = left_scaled[:, None, :] - right_scaled[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2)
    return np.exp(-0.5 * squared_distance)
