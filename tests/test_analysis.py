"""Scientific regression and control checks; no GPU, models, or network."""
import json
from pathlib import Path
import unittest

import numpy as np

from analyze_prompt_overlap import analyze as split_prompt_overlap, score_parts, summarize
from analyze_sae import analyze, cluster_bootstrap, report_pair
from rfm_agreement import overlap
from rfm_sae import split_examples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"


class Controls(unittest.TestCase):
    def test_empty_and_generic_overlap(self):
        self.assertEqual(overlap("", "volcano"), 0)
        items = [{"prompt_idx": p, "jlens": "shared", "nla": "shared"} for p in (0, 0, 1, 1)]
        result = report_pair(items, "jlens", "nla")
        self.assertEqual(result["excess"], 0)
        self.assertEqual(result["frac_above_own_null"], 0)
        np.testing.assert_allclose(cluster_bootstrap(items, "jlens", "nla")["excess_ci95"], [0, 0])

    def test_known_signal(self):
        items = [{"prompt_idx": p, "jlens": word, "nla": word}
                 for p, word in [(0, "volcano"), (0, "volcano"), (1, "mortgage"), (1, "mortgage")]]
        self.assertEqual(report_pair(items, "jlens", "nla")["excess"], 1)
        np.testing.assert_allclose(cluster_bootstrap(items, "jlens", "nla")["excess_ci95"], [1, 1])

    def test_prompt_only_and_mixed_agreement(self):
        np.testing.assert_allclose(score_parts({"volcano"}, {"volcano"}, {"volcano"}, {"volcano"}), [1, 1, 0, 0])
        np.testing.assert_allclose(score_parts({"volcano", "ash"}, {"volcano", "ash"}, {"volcano"}, {"volcano"}), [1, .5, .5, 1])

    def test_topic_only_signal_disappears(self):
        ids = np.array([0, 0, 1, 1])
        sets = [{"volcano"}, {"volcano"}, {"mortgage"}, {"mortgage"}]
        matrix = np.empty((4, 4, 4))
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                matrix[:, i, j] = score_parts(a, b, sets[i], {"volcano", "mortgage"})
        result = summarize(matrix, ids, [])
        self.assertEqual(result["scores"]["overall"]["excess"], 1)
        for component in ("non_source_prompt_contribution", "all_prompts_removed"):
            self.assertEqual(result["scores"][component]["excess"], 0)
            np.testing.assert_allclose(result["scores"][component]["excess_ci95"], [0, 0])

    def test_duplicate_documents_cannot_validate_labels(self):
        # Every document has identical text; there is no independent held-out window.
        rows = [{"doc_idx": i, "text": "Same example", "rank": i, "activation": 5-i} for i in range(5)]
        train, held = split_examples(rows)
        self.assertFalse(train)
        self.assertIsNone(held)


class RecordedResults(unittest.TestCase):
    def test_primary_results_reproduce(self):
        load = lambda name: json.loads((OUT / name).read_text())
        annotations = load("sae_review.json")
        annotations["initial_annotation_seconds"] = load("sae_labels.json")["elapsed_seconds"]
        computed = analyze(load("agreement.json"), load("sae_encode.json"), annotations)
        recorded = load("sae_comparisons.json")
        self.assertEqual(computed["report"], recorded["report"])
        self.assertEqual(computed["coverage"], recorded["coverage"])
        self.assertEqual(computed["prompt_cluster_bootstrap"], recorded["prompt_cluster_bootstrap"])
        self.assertFalse(computed["training_performed"])
        self.assertFalse(computed["steering_performed"])

    def test_prompt_control_reproduces(self):
        computed = split_prompt_overlap(json.loads((OUT / "sae_comparisons.json").read_text()),
                                        json.loads((OUT / "agreement.json").read_text()))
        recorded = json.loads((OUT / "prompt_overlap.json").read_text())
        self.assertEqual(computed["report"], recorded["report"])
        self.assertEqual([r["scores"]["overall"]["n"] for r in computed["report"].values()], [48, 44, 44])


if __name__ == "__main__":
    unittest.main()
