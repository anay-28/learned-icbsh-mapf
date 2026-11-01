"""Transparent JSON inference and a conservative residual override boundary."""
import json
from pathlib import Path
import time
import numpy as np

from mapf.features import FeatureExtractor
from mapf.ml_revision_policy import candidate_pool
from mapf.override_features import SYMMETRIC_FEATURE_NAMES, symmetric_features


class SavedOverride:
    def __init__(self, path):
        from cro_protocol import digest, sha
        path = Path(path)
        data = json.loads(path.read_text())
        metadata = json.loads(Path(str(path) + '.metadata.json').read_text())
        if metadata['model_sha256'] != sha(path):
            raise ValueError('Override model checksum mismatch')
        if data['model_sha256'] != digest({k: v for k, v in data.items() if k != 'model_sha256'}):
            raise ValueError('Override model payload checksum mismatch')
        if data['schema_version'] != 'bdro-model-v1' or data['feature_names'] != SYMMETRIC_FEATURE_NAMES:
            raise ValueError('Override feature schema mismatch')
        self.data = data
        self.backend, self.tau = data['backend'], data['tau']
        if self.backend not in ('logistic', 'ridge') or self.tau != .75:
            raise ValueError('Non-frozen backend or threshold')
        self.mean, self.scale, self.coef = [np.asarray(data[k], dtype=float) for k in ('mean', 'scale', 'coef')]
        self.intercept = float(data['intercept'])
        if any(v.shape != (24,) or not np.isfinite(v).all() for v in (self.mean, self.scale, self.coef)):
            raise ValueError('Invalid model dimensions/values')
        if (self.scale <= 0).any() or not np.isfinite(self.intercept):
            raise ValueError('Invalid model scale/intercept')

    def predict_probability(self, X):
        if self.backend != 'logistic':
            raise ValueError('Ridge diagnostic is not an override probability')
        z = ((np.asarray(X) - self.mean) / self.scale) @ self.coef + self.intercept
        # Stable sigmoid, also for logits with large magnitude.
        return np.exp(-np.logaddexp(0., -z))

    def predict(self, X):
        return ((np.asarray(X) - self.mean) / self.scale) @ self.coef + self.intercept


class ResidualOverrideSelector:
    def __init__(self, model, tau=.75):
        if tau != .75:
            raise ValueError('BDRO threshold is frozen at .75')
        self.model, self.tau, self.extractor = model, tau, None
        self.stats = dict(selector_calls=0, single_candidate=0, override_opportunities=0,
                          overrides=0, candidates_scored=0, pool_seconds=0.,
                          feature_seconds=0., inference_seconds=0.)

    def __call__(self, candidates, context):
        self.stats['selector_calls'] += 1
        began = time.perf_counter()
        pool = candidate_pool(candidates, context)
        self.stats['pool_seconds'] += time.perf_counter() - began
        if len(pool) == 1:
            self.stats['single_candidate'] += 1
            return pool[0]
        began = time.perf_counter()
        if self.extractor is None:
            self.extractor = FeatureExtractor(context['grid'], context['goals'])
        phi = symmetric_features(pool, context, self.extractor)
        self.stats['feature_seconds'] += time.perf_counter() - began
        began = time.perf_counter()
        probabilities = np.asarray(self.model.predict_probability(phi[1:] - phi[0]))
        self.stats['inference_seconds'] += time.perf_counter() - began
        if probabilities.shape != (len(pool) - 1,) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError('Invalid override probabilities')
        self.stats['override_opportunities'] += 1
        self.stats['candidates_scored'] += len(probabilities)
        best = int(np.argmax(probabilities))
        if probabilities[best] >= self.tau:
            self.stats['overrides'] += 1
            return pool[best + 1]
        return pool[0]
