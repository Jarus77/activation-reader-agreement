# Activation reader agreement on Qwen3.6-27B

**Do J-lens, natural-language autoencoder (NLA), and sparse autoencoder (SAE) readouts of the same activation agree more than readouts paired across different prompts? Does that agreement survive removing literal prompt vocabulary?**


**Finding:** all three pairs show positive excess agreement on this sample, including after prompt-vocabulary removal. The result is exploratory: topic-related words remain, SAE descriptions are automatically screened, and independent instrument validation is incomplete.

## Results

Twelve prompts × four token positions gave 48 activations. SAE rows use the 44 activations with at least one accepted feature description.

| Pair | n | Matched | Mismatched | Excess | z* | Beats all mispairings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J-lens vs NLA | 48 | 0.329 | 0.023 | +0.306 | +4.43 | 75% |
| NLA vs SAE | 44 | 0.208 | 0.052 | +0.156 | +2.42 | 32% |
| J-lens vs SAE | 44 | 0.143 | 0.009 | +0.134 | +2.79 | 43% |

**Matched** is mean overlap for the same activation; **mismatched** is mean overlap across all ordered pairs from different prompts. **Excess** subtracts the latter from the former. **z*** divides excess by the standard deviation of pairwise mismatch scores; it is a descriptive effect size, not a significance-test statistic. **Beats all mispairings** requires a matched score to strictly exceed every cross-prompt alternative: 44 alternatives per J-lens–NLA item and 40–42 per SAE item. Ties do not count.

![Excess agreement with prompt-cluster bootstrap intervals](artifacts/agreement_excess.png)

The intervals resample prompts while holding feature labels fixed. They do not account for annotation errors or uncertainty in the pretrained instruments.

### After removing prompt vocabulary

We removed the union of all 12 prompts' normalized words identically from both readouts, then recomputed overlap with the smaller remaining word set as denominator. Filtering-induced empty sets score zero; the original 48/44/44 item support is preserved.

| Pair | n | Matched | Mismatched | Remaining excess | z* | Beats all mispairings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J-lens vs NLA | 48 | 0.305 | 0.023 | +0.282 | +4.05 | 69% |
| NLA vs SAE | 44 | 0.196 | 0.053 | +0.143 | +2.16 | 32% |
| J-lens vs SAE | 44 | 0.120 | 0.009 | +0.111 | +2.33 | 36% |

The conditional 95% prompt-bootstrap intervals for remaining excess are **[+0.235, +0.331]**, **[+0.096, +0.190]**, and **[+0.059, +0.167]**, respectively.

A separate additive decomposition retains the original denominator and splits shared words by whether they appear in each source prompt. Words absent from that prompt account for **87.0%, 84.8%, and 79.2%** of the original excess, respectively. This decomposition and the re-normalized table above answer different questions.

Literal prompt-word copying alone does not explain the measured agreement. Topic synonyms and continuation structure could still explain part of it. These scores do not establish causal faithfulness or rank the methods' quality.

Further reading: [research summary](research_summary.md), [three-page report](artifacts/research_findings.pdf), [report source](research_findings.md), and [full prompt-word analysis](artifacts/prompt_overlap_report.md). Exportable tables: [original](artifacts/agreement_table_original.png) and [prompt vocabulary removed](artifacts/agreement_table_prompt_removed.png), also available as PDF and SVG.

## Experimental setup

**Source activations.** We used Qwen3.6-27B, with 64 blocks and residual width 5,120. Generation was deterministic, limited to 150 new tokens, with thinking disabled. We recorded the residual stream after zero-based block 42 (`hidden_states[43]`) at absolute token indices **60, 85, 110, 135 in the full chat sequence**. All three methods read the same saved vectors.

**J-lens.** A published Jacobian lens fitted on 1,000 prompts maps each activation through `J[42]` and the model's unembedding. We retain the 12 highest-scoring vocabulary tokens.

**NLA.** The published RL verbalizer uses its merged `av_base` warm-start weights and the `iter_000600` adapter. It receives the activation through additive, norm-matched injection at block 1 and generates a description with a 256-token limit. It receives no evaluation prompt or continuation as text. All 48 descriptions completed the expected format. This injection is part of the readout mechanism; the source model was not steered.

**SAE.** We used the pretrained, layer-compatible 2,097,152-feature SAE. No SAE or other model was trained. The published centered encoding is evaluated in float32:

```python
pre = (h - b_dec) @ W_enc.T + b_enc
acts = pre * (pre > threshold)  # recorded threshold: 1.682811975479126
```

We select the eight strongest active features per activation and concatenate accepted descriptions for the overlap analysis. Reconstruction explained variance on these 48 activations was **0.601**, a sample diagnostic rather than an independent population estimate.

**Feature descriptions.** A separate Qwen3-4B-Instruct model proposed labels from published maximum-activation windows. An entire document was withheld from label construction; checks used one withheld window and one random other-feature window, with order counterbalanced. Labels were accepted only when supported by the former and rejected by the latter. Exact-duplicate withheld text across document IDs excluded 18 otherwise accepted labels before SAE agreement scores were examined. A separate Qwen3.6-27B base-model review retained **88 of 154** screened candidate labels. Neither labeler nor reviewer received evaluation prompts or instrument readouts.

Final coverage was **44/48 activations, 105/384 selected feature slots, and 88/272 unique selected features**. Counting the four missing-label activations as zero gives original excess **+0.142** for NLA–SAE and **+0.122** for J-lens–SAE. The descriptions remain automatic and contain residual errors; see [label-quality notes](artifacts/sae_label_quality_notes.json).

**Scoring and uncertainty.** Readouts become English word sets after stop-word removal and light plural normalization. Overlap is `len(A & B) / min(len(A), len(B))`, or zero for an empty set. Same-prompt mismatches are excluded. We use 4,000 prompt-cluster bootstrap replicates, seed 1729, retaining each prompt's positions together and holding labels fixed. The prompt-word analyses are post hoc. Pairwise mismatches are dependent and are not treated as independent observations for confidence intervals.

## Prompts, verbatim

```text
Explain how a volcano erupts and what happens to the surrounding area.
Explain how a fixed-rate mortgage works and what affects the interest rate.
Explain the rules of cricket to someone who has never seen a match.
Describe what happens in a typical opera and why the form endures.
Explain how glaciers form and how they reshape a landscape.
Explain how mitochondria produce energy inside a cell.
Explain how earthquakes happen along tectonic fault lines.
Explain how a compiler turns source code into a running program.
Describe how migrating birds navigate over long distances.
Explain how coral reefs form and why they are bleaching.
Explain how bread dough rises and what yeast is doing.
Explain how a vaccine trains the immune system.
```

The saved [agreement readouts](artifacts/agreement.json) include generated continuations, token positions, J-lens tokens, and NLA descriptions. Label construction and review prompts are in [rfm_sae.py](rfm_sae.py), and their recorded inputs and outputs are in [sae_labels.json](artifacts/sae_labels.json) and [sae_review.json](artifacts/sae_review.json).

## Reproduce the analysis without a GPU

Use Python 3.11 or newer. No model downloads, Modal account, or upstream checkout is required for this path.

```sh
git clone https://github.com/Jarus77/activation-reader-agreement.git
cd activation-reader-agreement
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
python analyze_sae.py
python analyze_prompt_overlap.py
python render_agreement_tables.py
python render_findings.py
```

These commands reproduce the reported scores and bootstrap intervals from the saved readouts and regenerate the tables, figure, and three-page PDF. They rewrite derived analysis files, leaving source readouts intact. The tests include known-signal and null controls, duplicate-example screening, and regression checks against both recorded result tables. [manifest.json](artifacts/manifest.json) records source-data hashes, checkpoint revisions, and run IDs.

## Collect a fresh experiment on Modal

This path allocates paid compute and downloads approximately 150–170 GB of weights and examples into a persistent volume. Authentication is local; no credentials are included here. Complete each stage before starting the next; `build` can run while `warm` downloads.

```sh
python -m pip install -r requirements.txt
python -m modal token new
python modal_smoke.py metadata
python modal_smoke.py warm
python modal_smoke.py build
python modal_smoke.py contract
python modal_smoke.py gpu
python modal_smoke.py nla
python modal_smoke.py agreement
python modal_smoke.py sae_warm
python modal_smoke.py sae_encode
python modal_smoke.py sae_labels
python modal_smoke.py sae_review
python analyze_sae.py
python analyze_prompt_overlap.py
```

All Modal app, function, and volume names are **`rfm`**, with only the **`rfm=rfm`** tag. External trackers are disabled. Each function uses one container with no automatic retries and a two-second idle scale-down window. Source collection and review use an H200; the smaller annotation model has L4/A10/L40S/H200 fallbacks. Function timeout caps are 90 minutes for agreement, 30 minutes for smoke/NLA/annotation, and 20 minutes for SAE encoding/review. These are per-stage caps, not a total spending limit.

Recorded execution times were **35 minutes** for the main collection, **70 seconds** for SAE encoding, **190 seconds** for initial annotation, and **172 seconds** for review. Downloads, image builds, queues, and compatibility checks add time.

The remote image pins Python 3.11, PyTorch 2.8.0/CUDA 12.8, Transformers 5.5.4, PEFT 0.19.1, Accelerate 1.12.0, Safetensors 0.8.0, PyYAML 6.0.3, and PyArrow 26.0.0. It uses native PyTorch GDN and eager attention; FLA is excluded following a Hopper correctness guard encountered during preflight. The image installs Jacobian-lens at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

```sh
python modal_smoke.py status agreement
python modal_smoke.py progress agreement
python modal_smoke.py logs agreement
python modal_smoke.py retrieve agreement
python modal_smoke.py cancel agreement
```

Launch records are created locally for fresh runs; they are not part of this release. Detached calls can finish after the client disconnects. Activation tensors and checkpoints persist under `/cache/runs/<run_id>/` in the `rfm` volume. A successful new agreement collection archives prior SAE analysis data so it cannot silently mix old labels with new activations. Fresh label construction now excludes duplicate text before proposing labels; the recorded run applied that exclusion after initial annotation. A new collection therefore need not be byte-for-byte identical to the archived experiment.

### Immutable checkpoints

| Role | Hugging Face repository | Revision |
| --- | --- | --- |
| Source model | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| NLA warm start + RL adapter | [ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl) | `5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f` |
| J-lens weights | [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) | `9fa267f841d62aa468dab5b9b38c3dc45a82d863` |
| Block-42 SAE | [ceselder/qwen36-27b-sae2m-l42](https://huggingface.co/ceselder/qwen36-27b-sae2m-l42) | `3233a3d8744da53999fe5e6d36a8357b873330ba` |
| Label model | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | `cdbee75f17c01a7cc42f958dc650907174af0554` |

The NLA adapter must attach to its published merged warm-start base, rather than raw Qwen weights. SAE comparisons require block 42; the separately published block-43 full-stack checkpoint is excluded.

## Files and limits

| File | Purpose |
| --- | --- |
| `modal_smoke.py` | Pinned downloads, image construction, bounded launches, retrieval and cancellation |
| `rfm_runtime.py`, `rfm_agreement.py` | Compatibility checks, activation collection, J-lens and NLA readouts |
| `rfm_sae.py` | Pretrained SAE encoding, feature descriptions, duplicate screening and review |
| `analyze_sae.py`, `analyze_prompt_overlap.py` | Offline comparisons, prompt controls and cluster bootstrap |
| `render_agreement_tables.py`, `render_findings.py` | PNG/PDF/SVG tables, figure and report |
| `tests/test_analysis.py` | Offline scientific controls and recorded-result regression checks |
| `artifacts/` | Source readouts, derived results, figures and provenance manifest |

The experiment covers one model, one block, and 12 prompts. English lexical overlap misses synonyms and is influenced by readout length and vocabulary. Automatic SAE descriptions have incomplete coverage and residual errors; the SAE publisher also reports repeated training documents due to a loader defect. Independent NLA reconstruction and J-lens provenance/readout validation remain pending in this pipeline. The instruments share model-family dependencies, so correlated errors are possible.

Same-topic mismatches and prompt-only expected-concept removal have been proposed but not run. Oracle-lens, Skip-lens, majority-vote scoring, and causal steering are outside the completed experiment. **No steering experiment has been launched; it requires separate user consultation and explicit approval.**

Code is released under [Apache-2.0](LICENSE); attribution is in [NOTICE](NOTICE). Pretrained weights are not distributed and retain their own licenses. SAE example excerpts retain their source terms. The repository runs independently of the reference checkout, with adapted prompts and scoring functions credited to tri-lens. Companion reports, credentials, model caches, and launch records are excluded from publication.
