"""Meaningful model-contract checks; these synthetic fixtures are not research data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mapf.multisignal_model import (fit_pairwise, SavedMultisignal, save_model,
                                   load_model, set_threshold, MultisignalSelector)


def examples():
    rows = []
    for family in range(3):
        for index in range(12):
            label = index % 2
            rows.append(dict(pair_id=f"{family}/{index}", family_id=f"f{family}",
                             query_id=f"q{family}/{index}", split="train", ll_label=label,
                             hl_label=1-label, feature_difference=[(2*label-1)*(1+index/20), 0.0]))
    return rows


class MultisignalModelTests(unittest.TestCase):
    def test_saved_shared_score_is_antisymmetric_and_matches_candidate_difference(self):
        payload = fit_pairwise(examples(), ["signal", "constant"])
        self.assertTrue(payload["trained"])
        model = SavedMultisignal(payload)
        x = np.asarray([[2., 7.], [-1., 1.], [0.5, -2.]])
        d = x[1:] - x[0]
        np.testing.assert_allclose(model.scores(d), model.scores(x[1:])-model.scores(x[0]), atol=1e-12)
        np.testing.assert_allclose(model.predict_probability(d)+model.predict_probability(-d), 1., atol=1e-12)
        self.assertGreater(float(model.predict_probability([2., 0.])), .5)
        with tempfile.TemporaryDirectory() as directory:
            path = save_model(Path(directory)/"model.json", payload)
            np.testing.assert_array_equal(load_model(path).scores(x), model.scores(x))

    def test_auxiliary_never_overrides_resolved_primary_and_cannot_pass_support_gate(self):
        data = examples()
        primary = fit_pairwise(data, ["signal", "constant"])
        aux_no_unknown = fit_pairwise(data, ["signal", "constant"], "confirmed_aux")
        np.testing.assert_array_equal(primary["coef"], aux_no_unknown["coef"])
        unknowns = [dict(p, pair_id="aux/"+p["pair_id"], ll_label=None,
                        feature_difference=[-v for v in p["feature_difference"]]) for p in data]
        augmented = fit_pairwise(data+unknowns, ["signal", "constant"], "confirmed_aux")
        self.assertEqual(augmented["auxiliary_used_pairs"], len(unknowns))
        np.testing.assert_array_equal(primary["scale"], augmented["scale"])
        stopped = fit_pairwise(unknowns, ["signal", "constant"], "confirmed_aux")
        self.assertFalse(stopped["trained"])

    def test_protected_splits_nonfinite_and_duplicate_pairs_rejected(self):
        for split in ("validation", "test", "pilot"):
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                fit_pairwise([dict(examples()[0], split=split)], ["signal", "constant"])
        with self.assertRaisesRegex(ValueError, "Invalid feature"):
            fit_pairwise([dict(examples()[0], feature_difference=[float("nan"), 0])], ["signal", "constant"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fit_pairwise([examples()[0], examples()[0]], ["signal", "constant"])

    def test_family_replication_does_not_change_fit_or_scale(self):
        original = examples()
        replicated = original + [dict(p, pair_id="rep/"+p["pair_id"]) for p in original if p["family_id"]=="f0"]
        a = fit_pairwise(original, ["signal", "constant"])
        b = fit_pairwise(replicated, ["signal", "constant"])
        np.testing.assert_allclose(a["scale"], b["scale"], atol=1e-14)
        np.testing.assert_allclose(a["coef"], b["coef"], atol=1e-8)

    def test_tamper_and_feature_schema_detection(self):
        model = fit_pairwise(examples(), ["signal", "constant"])
        corrupted = copy.deepcopy(model)
        corrupted["coef"][0] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            SavedMultisignal(corrupted)
        with self.assertRaisesRegex(ValueError, "schema differs"):
            SavedMultisignal(model, ["constant", "signal"])
        selected = set_threshold(model, .8, {"split":"validation", "schedule_sha256":"fixture"})
        self.assertEqual(SavedMultisignal(selected).tau, .8)
        self.assertEqual(model["coef"], selected["coef"])
        with self.assertRaisesRegex(ValueError, "grid"):
            set_threshold(model, .75, {})

    def test_selector_uses_best_challenger_and_falls_back_without_valid_features(self):
        # Feature providers are mocked to isolate the actual maximum-probability gate.
        model = fit_pairwise(examples(), ["signal", "constant"])
        pool = [object(), object(), object()]
        with patch("mapf.ml_revision_policy.candidate_pool", return_value=pool), \
             patch("mapf.multisignal_teacher.FEATURE_NAMES", ["signal", "constant"]), \
             patch("mapf.multisignal_teacher.candidate_features", return_value=np.array([[0,0],[1,0],[3,0]])):
            selector = MultisignalSelector(model, tau=.6)
            self.assertIs(selector(pool, {}), pool[2])
            self.assertEqual(selector.stats["overrides"], 1)
            disabled = MultisignalSelector(model, tau=1.01)
            self.assertIs(disabled(pool, {}), pool[0])
            self.assertEqual(disabled.stats["candidates_scored"], 0)
        with patch("mapf.ml_revision_policy.candidate_pool", return_value=pool), \
             patch("mapf.multisignal_teacher.FEATURE_NAMES", ["signal", "constant"]), \
             patch("mapf.multisignal_teacher.candidate_features", side_effect=ValueError("missing")):
            selector = MultisignalSelector(model, tau=.6)
            self.assertIs(selector(pool, {}), pool[0])
            self.assertEqual(selector.stats["fallback_unsupported"], 1)


if __name__ == "__main__":
    unittest.main()
