"""Split saved agreement scores by prompt vocabulary; no model or remote calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rfm_agreement import norm_words

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
PAIRS = [("jlens", "nla"), ("nla", "sae"), ("jlens", "sae")]
NAMES = {"jlens|nla": "J-lens vs NLA", "nla|sae": "NLA vs SAE", "jlens|sae": "J-lens vs SAE"}
COMPONENTS = ["overall", "source_prompt_contribution", "non_source_prompt_contribution",
              "all_prompts_removed"]


def score_parts(a, b, source_prompt, all_prompt_words):
    denominator = min(len(a), len(b))
    common = a & b
    overall = len(common) / denominator if denominator else 0.0
    prompt = len(common & source_prompt) / denominator if denominator else 0.0
    nonprompt = len(common - source_prompt) / denominator if denominator else 0.0
    residual_a, residual_b = a - all_prompt_words, b - all_prompt_words
    residual_denominator = min(len(residual_a), len(residual_b))
    filtered = len(residual_a & residual_b) / residual_denominator if residual_denominator else 0.0
    return np.array([overall, prompt, nonprompt, filtered], dtype=float)


def summarize(matrices, prompt_ids, matched_word_counts, repetitions=4000):
    size = len(prompt_ids)
    off_prompt = np.not_equal.outer(prompt_ids, prompt_ids)
    groups = sorted(set(prompt_ids))
    group_indices = [np.flatnonzero(prompt_ids == p) for p in groups]
    counts = np.array([len(g) for g in group_indices], dtype=float)
    null_counts = np.outer(counts, counts)
    np.fill_diagonal(null_counts, 0)
    rng = np.random.default_rng(1729)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repetitions)
    denominators = np.einsum("bi,ij,bj->b", weights, null_counts, weights)
    eligible = denominators > 0
    rows = {}
    for component, matrix in zip(COMPONENTS, matrices):
        matched = np.diag(matrix)
        mismatched = matrix[off_prompt]
        matched_mean, null_mean = float(matched.mean()), float(mismatched.mean())
        null_sd = float(mismatched.std())
        grouped_matched = np.array([matched[g].sum() for g in group_indices])
        grouped_null = np.zeros_like(null_counts)
        for i, a in enumerate(group_indices):
            for j, b in enumerate(group_indices):
                if i != j:
                    grouped_null[i, j] = matrix[np.ix_(a, b)].sum()
        boot_matched = (weights @ grouped_matched) / (weights @ counts)
        boot_null = np.einsum("bi,ij,bj->b", weights[eligible], grouped_null, weights[eligible]) / denominators[eligible]
        boot_excess = boot_matched[eligible] - boot_null
        beats = [matched[i] > matrix[i, off_prompt[i]].max() for i in range(size)]
        rows[component] = {
            "n": size, "prompts": len(groups), "null_pairs": int(off_prompt.sum()),
            "matched": matched_mean, "mismatched": null_mean, "excess": matched_mean - null_mean,
            "null_sd": null_sd,
            "z_descriptive": (matched_mean - null_mean) / null_sd if null_sd > 1e-3 else None,
            "z_note": "Descriptive excess in pairwise null standard deviations; not a significance-test statistic",
            "excess_ci95": np.quantile(boot_excess, [.025, .975]).tolist(),
            "fraction_beats_all_mismatches": float(np.mean(beats)),
            "bootstrap_replicates": int(eligible.sum()), "bootstrap_seed": 1729,
        }
    total = rows["overall"]
    prompt = rows["source_prompt_contribution"]
    nonprompt = rows["non_source_prompt_contribution"]
    for key in ["matched", "mismatched", "excess"]:
        assert abs(total[key] - prompt[key] - nonprompt[key]) < 1e-12
    return {"scores": rows,
            "matched_overlap_prompt_fraction": prompt["matched"] / total["matched"] if total["matched"] else None,
            "excess_nonprompt_fraction": nonprompt["excess"] / total["excess"] if total["excess"] else None,
            "matched_word_counts": matched_word_counts}


def analyze(saved, agreement):
    assert saved["agreement_run_id"] == agreement["run_id"]
    assert saved["status"] == "exploratory_complete"
    prompts = agreement["protocol"]["prompts"]
    prompt_sets = [norm_words(p) for p in prompts]
    pooled = set().union(*prompt_sets)
    assert len({(i['prompt_idx'], i['pos']) for i in saved['items']}) == len(saved['items'])
    result = {
        "status": "exploratory_complete", "agreement_run_id": saved["agreement_run_id"],
        "annotation_run_id": saved["annotation_run_id"], "analysis": "post_hoc_prompt_vocabulary_split",
        "normalization": "Original English regex, stop words, and light plural normalization",
        "decomposition": "Intersection split by the left/source activation's actual input prompt; both components retain the original denominator and sum exactly to the original score. The source mask is fixed across its matched and mismatched candidates.",
        "removed_prompt_analysis": "Remove the union of normalized vocabulary from all 12 input prompts from both readouts, identically for every matched and mismatched pair; recompute the overlap denominator from remaining words.",
        "sample_policy": "Same pre-filter item support as original primary rows (48/44/44); word sets made empty by filtering stay in the analysis with score zero.",
        "uncertainty": "4,000 prompt-cluster bootstrap replicates, seed 1729, frozen labels; annotation and model uncertainty omitted",
        "prompts": [{"prompt_idx": i, "text": p, "normalized_words": sorted(prompt_sets[i])} for i,p in enumerate(prompts)],
        "pooled_prompt_vocabulary": sorted(pooled), "report": {}, "training_performed": False,
        "gpu_used": False, "steering_performed": False,
        "limitations": ["Literal word removal does not remove synonyms, semantic topic cues, or continuation templates",
                        "The decomposition's null labels are source-anchored; the pooled-removal analysis uses a uniform mask instead",
                        "Re-normalized filtered overlap is a different metric and should not be added to prompt contributions",
                        "Only 12 prompt clusters; selected topics, automatic SAE labels, and incomplete coverage",
                        "Post hoc sensitivity analysis, not a preregistered test or evidence of causality"],
    }
    for left, right in PAIRS:
        key = f"{left}|{right}"
        items = [i for i in saved["items"] if i[left] and i[right]]
        ids = np.array([i["prompt_idx"] for i in items])
        a_sets, b_sets = [norm_words(i[left]) for i in items], [norm_words(i[right]) for i in items]
        matrices = np.empty((len(COMPONENTS), len(items), len(items)))
        for i, a in enumerate(a_sets):
            for j, b in enumerate(b_sets):
                matrices[:, i, j] = score_parts(a, b, prompt_sets[ids[i]], pooled)
        word_counts = [{"prompt_idx": int(ids[i]), "pos": item["pos"],
                        "shared_prompt_words": sorted((a_sets[i] & b_sets[i]) & prompt_sets[ids[i]]),
                        "shared_nonprompt_words": sorted((a_sets[i] & b_sets[i]) - prompt_sets[ids[i]]),
                        "shared_words_after_pooled_removal": sorted((a_sets[i] & b_sets[i]) - pooled)}
                       for i,item in enumerate(items)]
        row = summarize(matrices, ids, word_counts)
        row["empty_readouts_after_pooled_removal"] = {
            "left": sum(not (s - pooled) for s in a_sets),
            "right": sum(not (s - pooled) for s in b_sets)}
        row["matched_positions_with_pooled_nonprompt_overlap"] = sum(bool(w["shared_words_after_pooled_removal"]) for w in word_counts)
        original = saved["report"][key]
        assert row["scores"]["overall"]["n"] == original["n"]
        for metric in ["matched", "mismatched", "excess"]:
            assert abs(row["scores"]["overall"][metric] - original[metric]) < 1e-12
        assert np.allclose(row["scores"]["overall"]["excess_ci95"], saved["prompt_cluster_bootstrap"][key]["excess_ci95"])
        result["report"][key] = row
    return result


def write_report(result):
    lines = ["# Does agreement survive removing prompt words?", "",
             "Post hoc analysis of the saved Qwen3.6-27B readouts. No GPU run, training, relabeling, or steering.", "",
             "## Split of the original score", "",
             "For each source activation, shared words are classified against its own input prompt. "
             "The same source-prompt mask is used for all its mismatch candidates. Prompt and non-prompt "
             "contributions share the original denominator, so they sum exactly to the original overlap "
             "before rounding. The displayed columns are rounded independently.", "",
             "| Pair | n | Matched: prompt | Matched: non-prompt | Null: prompt | Null: non-prompt | Excess: prompt | Excess: non-prompt |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, row in result["report"].items():
        p, n = row["scores"]["source_prompt_contribution"], row["scores"]["non_source_prompt_contribution"]
        lines.append(f"| {NAMES[key]} | {p['n']} | {p['matched']:.3f} | {n['matched']:.3f} | {p['mismatched']:.3f} | {n['mismatched']:.3f} | {p['excess']:+.3f} | {n['excess']:+.3f} |")
    lines += ["", "Non-prompt excess, 95% prompt-cluster intervals:", ""]
    for key,row in result["report"].items():
        s=row["scores"]["non_source_prompt_contribution"]
        lo,hi=s["excess_ci95"]
        lines.append(f"- {NAMES[key]}: {s['excess']:+.3f}, [{lo:+.3f}, {hi:+.3f}]; "
                     f"{row['excess_nonprompt_fraction']:.1%} of the original excess.")
    lines += ["", "## Remove all evaluation-prompt vocabulary", "",
              "A second analysis removes the union of normalized words in all 12 prompts from both readouts. "
              "This filter is identical for matched and mismatched pairs. Overlap is re-normalized using "
              "the remaining sets. Filtering-induced empty sets score zero; no activations are dropped.", "",
              "| Pair | n | Matched | Mismatched | Excess | z (descriptive) | Beats all mispairings | 95% excess interval |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key,row in result["report"].items():
        s=row["scores"]["all_prompts_removed"]
        lo,hi=s["excess_ci95"]
        z = f"{s['z_descriptive']:+.2f}" if s['z_descriptive'] is not None else "n/a"
        lines.append(f"| {NAMES[key]} | {s['n']} | {s['matched']:.3f} | {s['mismatched']:.3f} | {s['excess']:+.3f} | {z} | {s['fraction_beats_all_mismatches']:.0%} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "z is descriptive excess divided by the pairwise mismatch standard deviation, not a significance-test statistic. "
              "Beats all mispairings uses strict greater-than comparisons against every cross-prompt alternative."]
    lines += ["", "Filtered readout diagnostics:", ""]
    for key,row in result["report"].items():
        e=row["empty_readouts_after_pooled_removal"]
        n=row["scores"]["overall"]["n"]
        lines.append(f"- {NAMES[key]}: {e['left']} left and {e['right']} right readouts empty; "
                     f"{row['matched_positions_with_pooled_nonprompt_overlap']}/{n} matched pairs share a remaining word.")
    lines += ["", "## Interpretation", ""]
    positive = all(row["scores"]["all_prompts_removed"]["excess_ci95"][0] > 0 for row in result["report"].values())
    lines.append("All three pairs retain positive excess agreement after removing the full evaluation-prompt vocabulary, "
                 "with conditional bootstrap intervals above zero. Literal prompt-word overlap alone therefore "
                 "does not account for the observed agreement. " if positive else
                 "The filtered results and intervals above determine which pairs retain evidence of excess agreement. ")
    lines[-1] += "Synonyms, topic associations, and continuation structure remain possible explanations. "
    lines[-1] += "This does not establish token-specific information or causal accuracy."
    lines += ["", "## Limits and reproduction", ""]
    lines.extend(f"- {limit}" for limit in result["limitations"])
    lines += ["", "Run `python analyze_prompt_overlap.py`. The JSON preserves the vocabulary masks, "
              "per-position shared words, source-data hashes, run IDs, sample counts, and bootstrap settings.", ""]
    (OUT / "prompt_overlap_report.md").write_text("\n".join(lines))


def main():
    saved_path, agreement_path = OUT / "sae_comparisons.json", OUT / "agreement.json"
    saved = json.loads(saved_path.read_text())
    agreement = json.loads(agreement_path.read_text())
    result = analyze(saved, agreement)
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [saved_path, agreement_path]}
    (OUT / "prompt_overlap.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result)
    print((OUT / "prompt_overlap_report.md").read_text())


if __name__ == "__main__":
    main()
