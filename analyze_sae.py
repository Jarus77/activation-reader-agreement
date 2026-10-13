"""Join frozen SAE annotations with existing readouts and score both comparisons."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from rfm_agreement import norm_words, overlap

OUT = Path(__file__).resolve().parent / "artifacts"


def report_pair(items, left, right, include_empty=False):
    selected = list(items) if include_empty else [it for it in items if it[left] and it[right]]
    if len({it["prompt_idx"] for it in selected}) < 2:
        return {"status": "insufficient_valid_prompts", "n": len(selected)}
    matched = [overlap(it[left], it[right]) for it in selected]
    null = [overlap(a[left], b[right]) for a in selected for b in selected
            if a["prompt_idx"] != b["prompt_idx"]]
    m, mm, sd = st.mean(matched), st.mean(null), st.pstdev(null)
    counts, beats = [], []
    for a in selected:
        candidates = [overlap(a[left], b[right]) for b in selected
                      if a["prompt_idx"] != b["prompt_idx"]]
        counts.append(len(candidates))
        beats.append(overlap(a[left], a[right]) > max(candidates))
    return {"status": "exploratory", "n": len(selected), "attempted": len(items),
            "matched": m, "mismatched": mm, "excess": m - mm,
            "z": (m - mm) / sd if sd > 1e-3 else None,
            "frac_above_own_null": st.mean(beats), "null_pairs": len(null),
            "mispaired_candidates": counts, "nominal_rank_chance": st.mean(1 / (n + 1) for n in counts),
            "empty_left_content": sum(not norm_words(it[left]) for it in selected),
            "empty_right_content": sum(not norm_words(it[right]) for it in selected)}


def cluster_bootstrap(items, left, right, repetitions=4000):
    """Resample prompts; keep each prompt's positions together."""
    import numpy as np
    selected = [it for it in items if it[left] and it[right]]
    prompts = sorted({it["prompt_idx"] for it in selected})
    if len(prompts) < 2:
        return {"status": "insufficient_prompts"}
    groups = [[it for it in selected if it["prompt_idx"] == p] for p in prompts]
    counts = np.array([len(g) for g in groups], dtype=float)
    sums = np.array([sum(overlap(it[left], it[right]) for it in g) for g in groups])
    null_sums = np.zeros((len(groups), len(groups)))
    null_counts = np.outer(counts, counts)
    np.fill_diagonal(null_counts, 0)
    for i, a in enumerate(groups):
        for j, b in enumerate(groups):
            if i != j:
                null_sums[i, j] = sum(overlap(x[left], y[right]) for x in a for y in b)
    rng = np.random.default_rng(1729)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repetitions)
    matched = (weights @ sums) / (weights @ counts)
    denominator = np.einsum("bi,ij,bj->b", weights, null_counts, weights)
    eligible = denominator > 0
    mismatch = np.einsum("bi,ij,bj->b", weights[eligible], null_sums, weights[eligible]) / denominator[eligible]
    excess = matched[eligible] - mismatch
    return {"prompts": len(prompts), "replicates": int(eligible.sum()), "seed": 1729,
            "excess_ci95": np.quantile(excess, [0.025, 0.975]).tolist(),
            "note": "Prompt-cluster percentile bootstrap with frozen labels; annotation uncertainty is not included"}


def analyze(agreement, encoded, annotations):
    assert encoded["status"] == "encoding_complete"
    assert annotations["status"] == "annotation_complete"
    assert annotations["encoding_run_id"] == encoded["run_id"]
    assert agreement["run_id"] == encoded["agreement_run_id"]
    assert encoded["annotation_pack_sha256"] == annotations["annotation_pack_sha256"]
    normalized = lambda text: " ".join(text.casefold().split())
    duplicate_holdouts = {p["feature_id"] for p in encoded["annotation_pack"] if p["eligible"]
        and normalized(p["heldout_window"]["text"]) in
        {normalized(w["text"]) for w in p["training_windows"]}}
    accepted = {r["feature_id"]: r["label"] for r in annotations["labels"]
                if r["accepted"] and r["feature_id"] not in duplicate_holdouts}
    proposed = {r["feature_id"]: r["label"] for r in annotations["labels"] if r["format_valid"]}
    readouts = {(r["prompt_idx"], r["pos"]): r for r in agreement["readouts"]}
    assert len(readouts) == len(agreement["readouts"])
    items = []
    for item in encoded["items"]:
        r = readouts[(item["prompt_idx"], item["pos"])]
        assert r["format_complete"]
        features = [f["feature_id"] for f in item["features"]]
        items.append({"prompt_idx": item["prompt_idx"], "pos": item["pos"],
            "jlens": " ".join(r["jlens_tokens"]), "nla": r["explanation"],
            "sae": " ; ".join(accepted[f] for f in features if f in accepted),
            "sae_proposed": " ; ".join(proposed[f] for f in features if f in proposed),
            "features": item["features"], "accepted_feature_count": sum(f in accepted for f in features),
            "proposed_feature_count": sum(f in proposed for f in features)})
    report = {f"{left}|{right}": report_pair(items, left, right)
              for left, right in [("jlens", "nla"), ("nla", "sae"), ("jlens", "sae")]}
    return {"status": "exploratory_complete", "agreement_run_id": agreement["run_id"],
        "encoding_run_id": encoded["run_id"], "annotation_run_id": annotations["run_id"],
        "report": report, "sensitivity_include_empty_sae": {
            f"{left}|sae": report_pair(items, left, "sae", include_empty=True) for left in ("nla", "jlens")},
        "sensitivity_all_proposed_labels": {
            f"{left}|sae_proposed": report_pair(items, left, "sae_proposed") for left in ("nla", "jlens")},
        "prompt_cluster_bootstrap": {f"{left}|{right}": cluster_bootstrap(items, left, right)
            for left, right in [("jlens", "nla"), ("nla", "sae"), ("jlens", "sae")]},
        "coverage": {"items": len(items), "items_with_accepted_labels": sum(bool(i["sae"]) for i in items),
            "selected_feature_slots": sum(len(i["features"]) for i in items),
            "accepted_feature_slots": sum(i["accepted_feature_count"] for i in items),
            "unique_selected_features": encoded["unique_selected_features"],
            "unique_accepted_labels": len(accepted),
            "duplicate_holdout_features": len(duplicate_holdouts),
            "duplicate_holdout_labels_excluded": sum(r.get("initial_accepted", r["accepted"]) and r["feature_id"] in duplicate_holdouts
                                                       for r in annotations["labels"]),
            "accepted_labels_per_item": [i["accepted_feature_count"] for i in items]},
        "annotation_summary": annotations["summary"], "sae_reconstruction": encoded["checks"]["reconstruction"],
        "sae_repo": encoded["sae_repo"], "sae_revision": encoded["sae_revision"],
        "label_model": annotations["label_model"], "label_revision": annotations["label_revision"],
        "review_model": annotations.get("review_model"), "review_revision": annotations.get("review_revision"),
        "scoring_protocol_amendment": "Normalized exact-duplicate held-out windows were detected and excluded before any SAE comparison scores were inspected",
        "timings": {"encoding_seconds": encoded["elapsed_seconds"],
                    "annotation_seconds": annotations.get("initial_annotation_seconds", annotations["elapsed_seconds"]),
                    "review_seconds": annotations["elapsed_seconds"] if annotations.get("review_model") else 0},
        "training_performed": False, "steering_performed": False, "items": items,
        "limitations": [("Automatic SAE labels screened with a separate larger model; not independently human-validated semantics"
                         if annotations.get("review_model") else
                         "Automatic SAE labels and validation by the same small model; not independently human-validated semantics"),
            "SAE example-based labels were frozen without evaluation prompts, J-lens tokens, or NLA descriptions as input",
            "Each label checked on one withheld document and one random other-feature window",
            "Primary scoring excludes labels whose held-out window duplicates training-window text across different document IDs",
            "Feature coverage and proposed-label sensitivity are reported; unlabeled features cannot establish negative evidence",
            "The SAE publisher reports duplicated training documents; reconstruction here covers only 48 evaluation activations",
            "Original English lexical metric, not a semantic-equivalence measure",
            "z is a descriptive pairwise-null effect size, not a significance test; observations cluster by prompt",
            "Independent NLA reconstruction and published J-lens provenance validation remain pending"]}


def write_report(result):
    lines = ["# Qwen3.6-27B exploratory three-way agreement", "",
        "Measured on the same 48 layer-42 residual activations. SAE descriptions are automatically generated",
        "from published maximum-activation windows and screened on a withheld document plus a random-feature window.", "",
        "| Pair | n | Matched | Mismatched | Excess | z (descriptive) | Beats all mispairings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in [("jlens|nla", "J-lens vs NLA"), ("nla|sae", "NLA vs SAE"), ("jlens|sae", "J-lens vs SAE")]:
        row = result["report"][key]
        if "matched" not in row:
            lines.append(f"| {label} | {row['n']} | insufficient coverage | — | — | — | — |")
            continue
        z = "n/a" if row["z"] is None else f"{row['z']:+.2f}"
        lines.append(f"| {label} | {row['n']} | {row['matched']:.3f} | {row['mismatched']:.3f} | "
                     f"{row['excess']:+.3f} | {z} | {row['frac_above_own_null']:.2f} |")
    cov = result["coverage"]
    lines += ["", f"SAE label coverage: {cov['items_with_accepted_labels']}/{cov['items']} activations; "
        f"{cov['accepted_feature_slots']}/{cov['selected_feature_slots']} selected feature slots; "
        f"{cov['unique_accepted_labels']}/{cov['unique_selected_features']} unique selected features.", "",
        f"Duplicate-holdout check: {cov['duplicate_holdout_features']} features had repeated held-out text; "
        f"{cov['duplicate_holdout_labels_excluded']} otherwise accepted labels were excluded before scoring.", "",
        f"SAE reconstruction explained variance on this sample: {result['sae_reconstruction']['sample_explained_variance']:.3f}.", "",
        "The primary SAE rows use only screened labels and nonempty readouts, following the original missing-label policy.",
        "JSON results include all-items and all-proposed-label sensitivities.", "", "## Limits", ""]
    lines[-3:-3] = ["", "Prompt-cluster bootstrap 95% intervals for excess (labels held fixed):",
        *[f"- {pair}: {value['excess_ci95'][0]:+.3f} to {value['excess_ci95'][1]:+.3f}"
          for pair, value in result["prompt_cluster_bootstrap"].items() if "excess_ci95" in value], ""]
    lines += [f"- {note}" for note in result["limitations"]]
    quality_path = OUT / "sae_label_quality_notes.json"
    if quality_path.exists():
        lines += ["", "Spot checks still find occasional wording errors or unsupported qualifiers after automatic review. "
                  "See [label quality notes](sae_label_quality_notes.json)."]
    lines += ["", "No SAE training or steering was performed. Steering still requires explicit user approval.", ""]
    body = "\n".join(lines)
    (OUT / "sae_comparisons_report.md").write_text(body)
    (OUT / "agreement_report.md").write_text(body)


def main():
    agreement, encoded, annotations = (json.loads((OUT / name).read_text()) for name in
                                       ("agreement.json", "sae_encode.json", "sae_labels.json"))
    review_path = OUT / "sae_review.json"
    if review_path.exists():
        reviewed = json.loads(review_path.read_text())
        if reviewed.get("annotation_run_id") == annotations["run_id"] and reviewed.get("status") == "annotation_complete":
            reviewed.setdefault("initial_annotation_seconds", annotations["elapsed_seconds"])
            annotations = reviewed
    result = analyze(agreement, encoded, annotations)
    (OUT / "sae_comparisons.json").write_text(json.dumps(result, indent=2) + "\n")
    write_report(result)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("status", "report", "coverage", "annotation_summary", "timings")}, indent=2))


if __name__ == "__main__":
    main()
