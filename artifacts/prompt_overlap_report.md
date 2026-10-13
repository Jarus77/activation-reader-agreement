# Does agreement survive removing prompt words?

Post hoc analysis of the saved Qwen3.6-27B readouts. No GPU run, training, relabeling, or steering.

## Split of the original score

For each source activation, shared words are classified against its own input prompt. The same source-prompt mask is used for all its mismatch candidates. Prompt and non-prompt contributions share the original denominator, so they sum exactly to the original overlap before rounding. The displayed columns are rounded independently.

| Pair | n | Matched: prompt | Matched: non-prompt | Null: prompt | Null: non-prompt | Excess: prompt | Excess: non-prompt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| J-lens vs NLA | 48 | 0.040 | 0.289 | 0.000 | 0.023 | +0.040 | +0.267 |
| NLA vs SAE | 44 | 0.025 | 0.184 | 0.001 | 0.051 | +0.024 | +0.132 |
| J-lens vs SAE | 44 | 0.028 | 0.115 | 0.000 | 0.009 | +0.028 | +0.106 |

Non-prompt excess, 95% prompt-cluster intervals:

- J-lens vs NLA: +0.267, [+0.220, +0.316]; 87.0% of the original excess.
- NLA vs SAE: +0.132, [+0.092, +0.171]; 84.8% of the original excess.
- J-lens vs SAE: +0.106, [+0.061, +0.153]; 79.2% of the original excess.

## Remove all evaluation-prompt vocabulary

A second analysis removes the union of normalized words in all 12 prompts from both readouts. This filter is identical for matched and mismatched pairs. Overlap is re-normalized using the remaining sets. Filtering-induced empty sets score zero; no activations are dropped.

| Pair | n | Matched | Mismatched | Excess | z (descriptive) | Beats all mispairings | 95% excess interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| J-lens vs NLA | 48 | 0.305 | 0.023 | +0.282 | +4.05 | 69% | [+0.235, +0.331] |
| NLA vs SAE | 44 | 0.196 | 0.053 | +0.143 | +2.16 | 32% | [+0.096, +0.190] |
| J-lens vs SAE | 44 | 0.120 | 0.009 | +0.111 | +2.33 | 36% | [+0.059, +0.167] |

z is descriptive excess divided by the pairwise mismatch standard deviation, not a significance-test statistic. Beats all mispairings uses strict greater-than comparisons against every cross-prompt alternative.

Filtered readout diagnostics:

- J-lens vs NLA: 5 left and 0 right readouts empty; 42/48 matched pairs share a remaining word.
- NLA vs SAE: 0 left and 0 right readouts empty; 42/44 matched pairs share a remaining word.
- J-lens vs SAE: 3 left and 0 right readouts empty; 22/44 matched pairs share a remaining word.

## Interpretation

All three pairs retain positive excess agreement after removing the full evaluation-prompt vocabulary, with conditional bootstrap intervals above zero. Literal prompt-word overlap alone therefore does not account for the observed agreement. Synonyms, topic associations, and continuation structure remain possible explanations. This does not establish token-specific information or causal accuracy.

## Limits and reproduction

- Literal word removal does not remove synonyms, semantic topic cues, or continuation templates
- The decomposition's null labels are source-anchored; the pooled-removal analysis uses a uniform mask instead
- Re-normalized filtered overlap is a different metric and should not be added to prompt contributions
- Only 12 prompt clusters; selected topics, automatic SAE labels, and incomplete coverage
- Post hoc sensitivity analysis, not a preregistered test or evidence of causality

Run `python analyze_prompt_overlap.py`. The JSON preserves the vocabulary masks, per-position shared words, source-data hashes, run IDs, sample counts, and bootstrap settings.
