"""Fixed logistic override fitting, with train-only scaling and explicit STOP gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from cro_protocol import ROOT, artifact, digest, provenance, read_protocol, readlines, sha
from analyze_downstream import load_queries
from mapf.override_features import SYMMETRIC_FEATURE_NAMES
from mapf.override_policy import SavedOverride


def training_arrays(pairs):
    if any(p['split'] not in ('train', 'validation') or p['label'] not in (0, 1)
           or p['relation'] not in ('CHALLENGER_BETTER', 'RESIDUAL_BETTER') for p in pairs):
        raise ValueError('Only strict train/validation pairs are permitted')
    train = [p for p in pairs if p['split'] == 'train']
    if {p['family_id'] for p in train} & {p['family_id'] for p in pairs if p['split'] == 'validation'}:
        raise ValueError('Train/validation family leakage')
    if not train or {p['label'] for p in train} != {0, 1}:
        raise ValueError('Training needs both strict label classes')
    counts = Counter(p['query_id'] for p in train)
    X = np.asarray([p['feature_difference'] for p in train], dtype=float)
    if X.shape != (len(train), 24) or not np.isfinite(X).all():
        raise ValueError('Invalid training features')
    return train, X, np.asarray([p['label'] for p in train]), np.asarray([1 / counts[p['query_id']] for p in train])


def validation_report(queries, model, gates):
    from sklearn.metrics import roc_auc_score
    strict_y, strict_p, weights = [], [], []
    rows, accepted = [], []
    for query in queries:
        if query['split'] != 'validation':
            continue
        differences = np.asarray([p['feature_difference'] for p in query['pairs']])
        probabilities = model.predict_probability(differences)
        strict_count = sum(p['label'] is not None for p in query['pairs'])
        for p, probability in zip(query['pairs'], probabilities):
            rows.append(dict(query_id=query['query_id'], family_id=query['family_id'],
                             candidate_index=p['candidate_index'], probability=float(probability),
                             label=p['label'], relation=p['relation']))
            if p['label'] is not None:
                strict_y.append(p['label'])
                strict_p.append(float(probability))
                weights.append(1 / strict_count)
        index = int(np.argmax(probabilities))
        if probabilities[index] >= .75:
            p = query['pairs'][index]
            accepted.append(dict(query_id=query['query_id'], family_id=query['family_id'],
                                 candidate_index=p['candidate_index'], probability=float(probabilities[index]),
                                 label=p['label'], relation=p['relation']))
    auc = float(roc_auc_score(strict_y, strict_p, sample_weight=weights)) if len(set(strict_y)) == 2 else None
    precision = sum(p['label'] == 1 for p in accepted) / len(accepted) if accepted else None
    checks = dict(roc_auc=auc is not None and auc >= gates['validation_auc'],
                  precision=precision is not None and precision >= gates['validation_precision'],
                  accepted_overrides=len(accepted) >= gates['validation_overrides'],
                  accepted_families=len({p['family_id'] for p in accepted}) >= gates['validation_families'])
    return dict(roc_auc=auc, precision_at_tau=precision, tau=.75, accepted_overrides=len(accepted),
                accepted_families=len({p['family_id'] for p in accepted}),
                accepted_unknown=sum(p['label'] is None for p in accepted),
                precision_rule='Query-level highest-probability action; ties/unknown accepted actions count as unproven (not wins)',
                checks=checks, gate_passed=all(checks.values()), predictions=rows, accepted=accepted,
                test_used=False, validation_used_for_fit=False)


def train(root=ROOT):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    root = Path(root)
    p = read_protocol(root)
    quality = json.loads((root / 'quality.json').read_text())
    if not quality['integrity_passed'] or not quality['headroom_gate_passed'] or p['smoke']:
        decision = dict(**provenance(p), trained=False, reason='Frozen integrity/headroom gate did not pass; no forced fitting')
        artifact(root, 'training_decision.json', decision)
        print(json.dumps(decision, indent=2))
        return decision
    if sha(root / 'rollouts.jsonl') != quality['raw_rollouts_sha256']:
        raise ValueError('Evidence changed after verification')
    pairs = readlines(root / 'pairs.jsonl')
    train_pairs, X, y, weights = training_arrays(pairs)
    scaler = StandardScaler().fit(X, sample_weight=weights)
    Z = scaler.transform(X)
    predictions = {}
    for backend in ('logistic', 'ridge'):
        estimator = (LogisticRegression(penalty='l2', C=1., fit_intercept=True, solver='liblinear',
                                       max_iter=10000, random_state=71) if backend == 'logistic' else
                     Ridge(alpha=10, fit_intercept=False))
        estimator.fit(Z, y if backend == 'logistic' else y * 2 - 1, sample_weight=weights)
        data = dict(schema_version='bdro-model-v1', backend=backend, feature_names=SYMMETRIC_FEATURE_NAMES,
                    mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=estimator.coef_.reshape(-1).tolist(),
                    intercept=float(np.asarray(estimator.intercept_).reshape(-1)[0]), tau=.75,
                    training_family_ids=sorted({r['family_id'] for r in train_pairs}),
                    validation_family_ids=p['families']['validation'],
                    training_query_count=len({r['query_id'] for r in train_pairs}),
                    positive_query_count=len({r['query_id'] for r in train_pairs if r['label'] == 1}),
                    protocol_sha256=p['protocol_sha256'], training_data_sha256=digest(train_pairs),
                    generation_source_sha256=p['source_sha256'], input_sha256=p['input_sha256'],
                    query_weighting='Every strict training query sums to one; weighted train-only pair scaler',
                    validation_used_for_fit=False, test_used=False)
        data['model_sha256'] = digest(data)
        path = artifact(root, f'model_{backend}.json', data)
        artifact(root, f'model_{backend}.json.metadata.json', dict(**provenance(p), backend=backend,
                 model_sha256=sha(path), training_data_sha256=digest(train_pairs),
                 validation_used_for_fit=False, test_used=False))
        loaded = SavedOverride(path)
        expected = estimator.predict_proba(Z)[:, 1] if backend == 'logistic' else estimator.predict(Z)
        actual = loaded.predict_probability(X) if backend == 'logistic' else loaded.predict(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        predictions[backend] = dict(maximum_serialization_error=float(np.max(np.abs(actual - expected))),
                                    values=actual.tolist())
    artifact(root, 'training_predictions.json', dict(**provenance(p), training_data_sha256=digest(train_pairs),
             pair_ids=[r['pair_sha256'] for r in train_pairs], backends=predictions))
    validation = dict(**provenance(p), model_sha256=sha(root / 'model_logistic.json'),
                      **validation_report(load_queries(root, quality['stage']), SavedOverride(root / 'model_logistic.json'), p['gates']))
    artifact(root, 'validation.json', validation)
    decision = dict(**provenance(p), trained=True, deployment_gate_passed=validation['gate_passed'],
                    primary='logistic', diagnostic='ridge', tau=.75)
    artifact(root, 'training_decision.json', decision)
    print(json.dumps({k: v for k, v in validation.items() if k not in ('predictions', 'accepted')}, indent=2))
    return decision


def verify_models(root, p):
    from sklearn.preprocessing import StandardScaler
    root = Path(root)
    train_pairs, X, _, weights = training_arrays(readlines(root / 'pairs.jsonl'))
    reference = StandardScaler().fit(X, sample_weight=weights)
    stored = json.loads((root / 'training_predictions.json').read_text())
    if stored['training_data_sha256'] != digest(train_pairs) or stored['pair_ids'] != [r['pair_sha256'] for r in train_pairs]:
        raise ValueError('Training data/prediction mismatch')
    for backend in ('logistic', 'ridge'):
        model = SavedOverride(root / f'model_{backend}.json')
        d = model.data
        if (d['protocol_sha256'] != p['protocol_sha256'] or d['training_data_sha256'] != digest(train_pairs)
                or d['training_family_ids'] != sorted({r['family_id'] for r in train_pairs})
                or d['validation_used_for_fit'] or d['test_used']):
            raise ValueError('Model provenance or split audit failed')
        np.testing.assert_array_equal(model.mean, reference.mean_)
        np.testing.assert_array_equal(model.scale, reference.scale_)
        actual = model.predict_probability(X) if backend == 'logistic' else model.predict(X)
        np.testing.assert_array_equal(actual, stored['backends'][backend]['values'])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--primary', choices=['logistic'], default='logistic')
    parser.add_argument('--tau', type=float, choices=[.75], default=.75)
    parser.add_argument('--seed', type=int, choices=[71], default=71)
    args = parser.parse_args()
    train(args.root)


if __name__ == '__main__':
    main()
