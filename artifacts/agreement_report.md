# Qwen3.6-27B exploratory three-way agreement

Measured on the same 48 layer-42 residual activations. SAE descriptions are automatically generated
from published maximum-activation windows and screened on a withheld document plus a random-feature window.

| Pair | n | Matched | Mismatched | Excess | z (descriptive) | Beats all mispairings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J-lens vs NLA | 48 | 0.329 | 0.023 | +0.306 | +4.43 | 0.75 |
| NLA vs SAE | 44 | 0.208 | 0.052 | +0.156 | +2.42 | 0.32 |
| J-lens vs SAE | 44 | 0.143 | 0.009 | +0.134 | +2.79 | 0.43 |

SAE label coverage: 44/48 activations; 105/384 selected feature slots; 88/272 unique selected features.

Duplicate-holdout check: 20 features had repeated held-out text; 18 otherwise accepted labels were excluded before scoring.

SAE reconstruction explained variance on this sample: 0.601.

The primary SAE rows use only screened labels and nonempty readouts, following the original missing-label policy.
JSON results include all-items and all-proposed-label sensitivities.

Prompt-cluster bootstrap 95% intervals for excess (labels held fixed):
- jlens|nla: +0.257 to +0.354
- nla|sae: +0.104 to +0.211
- jlens|sae: +0.078 to +0.189


## Limits

- Automatic SAE labels screened with a separate larger model; not independently human-validated semantics
- SAE example-based labels were frozen without evaluation prompts, J-lens tokens, or NLA descriptions as input
- Each label checked on one withheld document and one random other-feature window
- Primary scoring excludes labels whose held-out window duplicates training-window text across different document IDs
- Feature coverage and proposed-label sensitivity are reported; unlabeled features cannot establish negative evidence
- The SAE publisher reports duplicated training documents; reconstruction here covers only 48 evaluation activations
- Original English lexical metric, not a semantic-equivalence measure
- z is a descriptive pairwise-null effect size, not a significance test; observations cluster by prompt
- Independent NLA reconstruction and published J-lens provenance validation remain pending

Spot checks still find occasional wording errors or unsupported qualifiers after automatic review. See [label quality notes](sae_label_quality_notes.json).

No SAE training or steering was performed. Steering still requires explicit user approval.
