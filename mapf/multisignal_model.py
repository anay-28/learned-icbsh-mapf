"""Transparent shared candidate scorer for the separately frozen multi-work study.

LL expansions are the primary compute proxy, not runtime ground truth. Strict HL
preferences are optional low-weight auxiliary supervision only where LL evidence
is unresolved. No future rollout quantity is an inference feature.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


MODEL_SCHEMA = "multisignal-shared-logistic-v1"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _check_pairs(pairs, feature_names):
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be nonempty and unique")
    seen = set()
    for pair in pairs:
        if pair.get("split") != "train":
            raise ValueError("Only TRAIN evidence may enter model fitting or scaling")
        if not pair.get("family_id") or not pair.get("query_id"):
            raise ValueError("Family and query identifiers are required")
        if pair.get("ll_label") not in (0, 1, None) or pair.get("hl_label") not in (0, 1, None):
            raise ValueError("Labels must be strict binary preferences or null")
        x = np.asarray(pair["feature_difference"], dtype=float)
        if x.shape != (len(feature_names),) or not np.isfinite(x).all():
            raise ValueError("Invalid feature difference")
        if pair.get("pair_id") is not None:
            if pair["pair_id"] in seen:
                raise ValueError("Duplicate pair_id")
            seen.add(pair["pair_id"])


def fit_pairwise(pairs, feature_names, variant="confirmed", aux_weight=0.2,
                 min_direction_pairs=10, min_families=3, provenance=None):
    """Fit train-only s(c)-s(r) with fixed L2 regularization and no intercept.

    Pair fields: split, family_id, query_id, feature_difference, ll_label,
    hl_label; optional pair_id. Label 1 means challenger uses less observed work.
    Both ablations use a scaler fitted only to confirmed LL training pairs.
    Each confirmed family has equal total weight. For the auxiliary ablation,
    each family's strict HL-only pairs jointly receive ``aux_weight``; these
    weights are deliberately not renormalized to make auxiliary-only families
    as influential as families with primary evidence.
    """
    if variant not in ("confirmed", "confirmed_aux"):
        raise ValueError("Unknown supervision ablation")
    if not 0 < aux_weight < 1:
        raise ValueError("aux_weight must be strictly between zero and one")
    if min_direction_pairs < 1 or min_families < 1:
        raise ValueError("Support gates must be positive")
    pairs = list(pairs)
    feature_names = list(feature_names)
    _check_pairs(pairs, feature_names)
    confirmed = [p for p in pairs if p.get("ll_label") is not None]
    auxiliary = [p for p in pairs if p.get("ll_label") is None and p.get("hl_label") is not None]
    classes = Counter(p["ll_label"] for p in confirmed)
    class_families = {str(label): sorted({p["family_id"] for p in confirmed if p["ll_label"] == label})
                     for label in (0, 1)}
    base = dict(schema=MODEL_SCHEMA, variant=variant, feature_names=feature_names,
                training_data_sha256=digest(pairs), provenance=provenance or {},
                training_family_ids=sorted({p["family_id"] for p in pairs}),
                confirmed_pairs=len(confirmed), auxiliary_available_pairs=len(auxiliary),
                confirmed_class_counts={str(k): classes[k] for k in (0, 1)},
                confirmed_class_family_ids=class_families,
                min_direction_pairs=min_direction_pairs, min_families_per_direction=min_families,
                aux_weight=aux_weight, tau=1.01, threshold_selected=False,
                validation_used_for_fit=False, test_used=False,
                supervision="Strict LL-work preferences; optional HL-work preferences where LL unresolved",
                objective_caveat="Work supervision is a compute proxy, not evidence of runtime improvement")
    gate = (all(classes[k] >= min_direction_pairs for k in (0, 1)) and
            all(len(class_families[str(k)]) >= min_families for k in (0, 1)))
    if not gate:
        base.update(trained=False, reason="Insufficient confirmed LL support in both directions and families",
                    auxiliary_used_pairs=0)
        base["model_sha256"] = digest(base)
        return base

    confirmed_counts = Counter(p["family_id"] for p in confirmed)
    primary_weights = np.asarray([1 / confirmed_counts[p["family_id"]] for p in confirmed])
    primary_x = np.asarray([p["feature_difference"] for p in confirmed], dtype=float)
    # Zero-centered weighted RMS preserves s(c)-s(r) and reverse-pair symmetry.
    scale = np.sqrt(np.average(primary_x ** 2, axis=0, weights=primary_weights))
    scale[scale < 1e-12] = 1.0
    selected = confirmed + (auxiliary if variant == "confirmed_aux" else [])
    aux_counts = Counter(p["family_id"] for p in auxiliary)
    weights = list(primary_weights)
    if variant == "confirmed_aux":
        weights.extend(aux_weight / aux_counts[p["family_id"]] for p in auxiliary)
    x = np.asarray([p["feature_difference"] for p in selected], dtype=float)
    y = np.asarray([p["ll_label"] if p.get("ll_label") is not None else p["hl_label"] for p in selected])
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    estimator = LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs",
                                   max_iter=10000, tol=1e-10, random_state=71)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(x / scale, y, sample_weight=np.asarray(weights))
    base.update(trained=True, scale=scale.tolist(), coef=estimator.coef_[0].tolist(),
                intercept=0.0, C=1.0, solver="lbfgs", fit_intercept=False,
                scale_rule="Zero-centered weighted RMS of confirmed LL training differences only",
                weighting="Confirmed: total 1 per family; auxiliary: total aux_weight per family",
                selected_pairs=len(selected), auxiliary_used_pairs=len(selected)-len(confirmed),
                training_query_count=len({p["query_id"] for p in selected}),
                fit_iterations=int(estimator.n_iter_[0]))
    base["model_sha256"] = digest(base)
    return base


def save_model(path, model):
    """Write a verified JSON model; caller owns experiment-root immutability."""
    SavedMultisignal(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


class SavedMultisignal:
    def __init__(self, source, feature_names=None):
        self.data = (json.loads(Path(source).read_text(encoding="utf-8"))
                     if isinstance(source, (str, Path)) else dict(source))
        d = self.data
        unhashed = {k: v for k, v in d.items() if k != "model_sha256"}
        if d.get("schema") != MODEL_SCHEMA or digest(unhashed) != d.get("model_sha256"):
            raise ValueError("Model schema or checksum mismatch")
        self.feature_names = list(d["feature_names"])
        if feature_names is not None and self.feature_names != list(feature_names):
            raise ValueError("Inference feature schema differs from training")
        if d.get("validation_used_for_fit") or d.get("test_used"):
            raise ValueError("Model declares protected-data leakage")
        self.trained = bool(d["trained"])
        self.tau = float(d.get("tau", 1.01))
        self.scale = self.coef = None
        if self.trained:
            self.scale = np.asarray(d["scale"], dtype=float)
            self.coef = np.asarray(d["coef"], dtype=float)
            shape = (len(self.feature_names),)
            if (self.scale.shape != shape or self.coef.shape != shape or
                    not np.isfinite(self.scale).all() or not np.isfinite(self.coef).all() or
                    (self.scale <= 0).any() or d.get("intercept") != 0 or d.get("fit_intercept") is not False):
                raise ValueError("Invalid shared-scoring parameters")

    def scores(self, features):
        if not self.trained:
            raise ValueError("No fitted model; use residual fallback")
        x = np.asarray(features, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError("Invalid inference features")
        return (x / self.scale) @ self.coef

    def predict_probability(self, differences):
        logits = np.clip(self.scores(differences), -700, 700)
        return 1.0 / (1.0 + np.exp(-logits))


def load_model(path, feature_names=None):
    return SavedMultisignal(path, feature_names=feature_names)


def set_threshold(model, tau, validation_provenance):
    """Seal a parent-selected validation threshold without changing fitted weights."""
    SavedMultisignal(model)
    if tau not in THRESHOLDS:
        raise ValueError("Threshold must be from the frozen grid")
    result = {k: v for k, v in model.items() if k != "model_sha256"}
    result.update(tau=float(tau), threshold_selected=True,
                  validation_threshold_provenance=validation_provenance)
    result["model_sha256"] = digest(result)
    return result


class MultisignalSelector:
    """One highest-scoring challenger, with a residual fallback and measured overhead."""
    def __init__(self, model, tau=None):
        self.model = model if isinstance(model, SavedMultisignal) else SavedMultisignal(model)
        self.tau = self.model.tau if tau is None else float(tau)
        if self.tau not in THRESHOLDS:
            raise ValueError("Threshold must be from the frozen grid")
        self.stats = dict(selector_calls=0, candidate_choices=0, overrides=0,
                          fallback_unsupported=0, candidates_scored=0,
                          pool_seconds=0.0, feature_seconds=0.0, inference_seconds=0.0)

    def __call__(self, candidates, context):
        from mapf.ml_revision_policy import candidate_pool
        from mapf.multisignal_teacher import candidate_features, FEATURE_NAMES
        self.stats["selector_calls"] += 1
        began = time.perf_counter()
        pool = candidate_pool(candidates, context)
        self.stats["pool_seconds"] += time.perf_counter() - began
        if len(pool) < 2 or not self.model.trained or self.tau > 1:
            return pool[0]
        self.stats["candidate_choices"] += 1
        if list(FEATURE_NAMES) != self.model.feature_names:
            self.stats["fallback_unsupported"] += 1
            return pool[0]
        began = time.perf_counter()
        try:
            x = np.asarray(candidate_features(pool, context), dtype=float)
            if x.shape != (len(pool), len(FEATURE_NAMES)) or not np.isfinite(x).all():
                raise ValueError("Unsupported feature values")
        except (ValueError, KeyError, TypeError, IndexError, OverflowError):
            self.stats["fallback_unsupported"] += 1
            return pool[0]
        finally:
            self.stats["feature_seconds"] += time.perf_counter() - began
        began = time.perf_counter()
        probabilities = self.model.predict_probability(x[1:] - x[0])
        index = int(np.argmax(probabilities)) + 1
        probability = float(probabilities[index - 1])
        self.stats["inference_seconds"] += time.perf_counter() - began
        self.stats["candidates_scored"] += len(pool)
        if probability >= self.tau and probability > 0.5:
            self.stats["overrides"] += 1
            return pool[index]
        return pool[0]
