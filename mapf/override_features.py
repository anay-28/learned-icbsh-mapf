"""24-dimensional agent-swap invariant wrapper. Historical Model B is unchanged."""
import numpy as np
from mapf.features import FEATURE_NAMES, FeatureExtractor

ORDERED_PAIRS = [('corridor_score_i', 'corridor_score_j'),
                 ('local_density_i', 'local_density_j'), ('goal_dist_i', 'goal_dist_j'),
                 ('cardinal_degree_i', 'cardinal_degree_j'),
                 ('downstream_conflict_count_i', 'downstream_conflict_count_j'),
                 ('agent_i_path_length', 'agent_j_path_length'), ('delta_g_i', 'delta_g_j')]
SYMMETRIC_FEATURE_NAMES = list(FEATURE_NAMES)
for left, right in ORDERED_PAIRS:
    SYMMETRIC_FEATURE_NAMES[FEATURE_NAMES.index(left)] = left[:-2] + '_pair_min'
    SYMMETRIC_FEATURE_NAMES[FEATURE_NAMES.index(right)] = left[:-2] + '_pair_max'


def symmetrize(raw, feature_names=FEATURE_NAMES):
    if list(feature_names) != FEATURE_NAMES:
        raise ValueError('Raw feature schema mismatch')
    x = np.asarray(raw, dtype=np.float64)
    if x.shape[-1:] != (24,) or not np.isfinite(x).all():
        raise ValueError('Expected finite 24-dimensional features')
    result = x.copy()
    for left, right in ORDERED_PAIRS:
        i, j = FEATURE_NAMES.index(left), FEATURE_NAMES.index(right)
        result[..., i] = np.minimum(x[..., i], x[..., j])
        result[..., j] = np.maximum(x[..., i], x[..., j])
    return result


def symmetric_features(candidates, context, extractor=None):
    extractor = extractor or FeatureExtractor(context['grid'], context['goals'])
    return symmetrize(extractor.extract_all(candidates, context))
