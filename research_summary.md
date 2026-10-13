# Agreement between interpretability methods on Qwen3.6-27B

## Research question

**Do J-lens, NLA, and SAE readouts of the same Qwen3.6-27B activation agree more than readouts paired across different prompts? Does that agreement survive removing literal prompt vocabulary?**

We adapted the agreement experiment from [Morgan Hough’s tri-lens study](https://github.com/m9h/tri-lens). The motivation is to test whether different interpretability methods recover overlapping content from a representation. Agreement provides evidence of convergence, while its relationship to faithful explanation of model behavior remains a separate question.

## Methods and experiments

**Activation collection.** We used 12 English prompts covering different topics, with deterministic generation and thinking disabled. One example was: “Explain how a vaccine trains the immune system.” We recorded post-block-42 residual activations at absolute token indices 60, 85, 110, and 135 in the complete chat sequence, giving **48 activations**.

**Three readouts.** Each method received the same saved activation. J-lens produced 12 high-scoring vocabulary tokens through a pretrained Jacobian lens. A pretrained NLA verbalizer generated a natural-language description without receiving the evaluation prompt or continuation as text. A pretrained layer-42 SAE identified active features; we selected the eight strongest and combined their accepted descriptions. No new model or SAE was trained.

**SAE descriptions.** A separate Qwen3-4B model proposed labels from published maximum-activation examples. Checks used a withheld document and a random other-feature window; duplicated withheld text was excluded. A subsequent Qwen3.6-27B base-model review retained **88 labels**, covering **44 of 48 activations** and **105 of 384 selected feature slots**. Labeling and review received no evaluation prompts or instrument readouts. The labels remain automatically generated and screened.

**Agreement and controls.** We measured English word-set overlap after stop-word removal and light plural normalization: shared words divided by the smaller readout’s word-set size. Matched readouts came from the same activation. The null paired readouts across different prompts; same-prompt pairs were excluded. Excess agreement was the matched mean minus the mismatched mean.

**Prompt-word analysis.** We first split the original overlap into shared words present and absent in the source activation’s input prompt, keeping the original denominator. We then removed the combined normalized vocabulary of all 12 prompts from both readouts and recomputed overlap. This second filter was identical for matched and mismatched pairs. Activations whose word sets became empty remained in the analysis with zero overlap.

**Uncertainty.** We used 4,000 prompt-cluster bootstrap replicates, resampling the 12 prompts while keeping each prompt’s four activations together. Labels were held fixed, so the intervals reflect prompt-sampling uncertainty rather than annotation uncertainty.

## Results

Before removing prompt vocabulary:

| Pair | n | Matched | Mismatched | Excess | z* | Beats all mispairings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J-lens vs NLA | 48 | 0.329 | 0.023 | +0.306 | +4.43 | 75% |
| NLA vs SAE | 44 | 0.208 | 0.052 | +0.156 | +2.42 | 32% |
| J-lens vs SAE | 44 | 0.143 | 0.009 | +0.134 | +2.79 | 43% |

*z is a descriptive effect size in mismatch standard deviations, not a significance-test statistic. “Beats all mispairings” requires the matched score to exceed every available cross-prompt alternative.*

Words absent from each source prompt accounted for **87.0% of the original excess for J-lens–NLA, 84.8% for NLA–SAE, and 79.2% for J-lens–SAE**.

After removing vocabulary from all 12 prompts and re-normalizing:

| Pair | n | Matched | Mismatched | Remaining excess | 95% bootstrap interval |
| --- | ---: | ---: | ---: | ---: | --- |
| J-lens vs NLA | 48 | 0.305 | 0.023 | +0.282 | +0.235 to +0.331 |
| NLA vs SAE | 44 | 0.196 | 0.053 | +0.143 | +0.096 to +0.190 |
| J-lens vs SAE | 44 | 0.120 | 0.009 | +0.111 | +0.059 to +0.167 |

All three pairs retained positive excess, with conditional bootstrap intervals above zero. J-lens–NLA showed the largest measured excess in both analyses. SAE reconstruction explained variance was **0.601** on these 48 activations; this was a sample diagnostic.

## Conclusions and limits

**The methods show convergent agreement on these Qwen activations, and literal prompt-word overlap alone does not explain the finding.** This is consistent with some shared readable content being recovered by the instruments.

The experiment does not yet establish that this content is specific to individual token positions or causally responsible for generated behavior. Topic synonyms and continuation structure remain plausible explanations. Readouts differ in length and vocabulary, so their scores cannot rank instrument quality.

The study is exploratory: it uses 12 prompts, incomplete SAE label coverage, automatic labels with residual errors, and an English lexical metric. The SAE publisher documents duplicated training data; independent NLA reconstruction and J-lens provenance/readout validation remain pending. The models also share the Qwen family, so the evidence is not from fully independent systems.

Same-topic mismatched prompts and prompt-only expected-concept removal are proposed next controls; neither has been run. Causal steering has not been performed.

Recorded results: [original comparisons](artifacts/sae_comparisons.json) and [prompt-word analysis](artifacts/prompt_overlap.json). The scripts and checkpoint details are documented in [README](README.md).
