"""Qwen agreement data collection and original cross-prompt scoring.

Scores are exploratory until independent instrument validation is complete.
The reference z is a descriptive null-standard-deviation effect size, not a
hypothesis-test statistic. Missing/invalid outputs are reported explicitly.
"""
from __future__ import annotations
import gc
import json
import statistics as st
import time
import traceback
from pathlib import Path

POSITIONS = [60, 85, 110, 135]

PROMPTS = [
 "Explain how a volcano erupts and what happens to the surrounding area.",
 "Explain how a fixed-rate mortgage works and what affects the interest rate.",
 "Explain the rules of cricket to someone who has never seen a match.",
 "Describe what happens in a typical opera and why the form endures.",
 "Explain how glaciers form and how they reshape a landscape.",
 "Explain how mitochondria produce energy inside a cell.",
 "Explain how earthquakes happen along tectonic fault lines.",
 "Explain how a compiler turns source code into a running program.",
 "Describe how migrating birds navigate over long distances.",
 "Explain how coral reefs form and why they are bleaching.",
 "Explain how bread dough rises and what yeast is doing.",
 "Explain how a vaccine trains the immune system.",
]

STOP = set("the a an of to and or in on at for with is are was were be been being that this these those it its as by from about into than then there their they which who what when where how not no but if so such can may text vector activation token tokens concept explanation snippet snippets describing describes related relating references referring semantic content model language word words context usage term likely signals pattern format structure structured establishing begins final".split())

def norm_words(s):
    import re
    out=set()
    for w in re.findall(r"[a-z]+", s.lower()):
        if len(w)<3 or w in STOP: continue
        if len(w)>4 and w.endswith("es"): out.add(w[:-2])
        if len(w)>3 and w.endswith("s"): out.add(w[:-1])
        out.add(w)
    return out

def overlap(a,b):
    A=norm_words(" ".join(a) if isinstance(a,list) else a)
    B=norm_words(" ".join(b) if isinstance(b,list) else b)
    if not A or not B: return 0.0
    return len(A&B)/min(len(A),len(B))


def report(items):
    valid = [it for it in items if it.get("explanation") and it.get("format_complete")]
    if len({it["prompt_idx"] for it in valid}) < 2:
        return {"status": "insufficient_valid_prompts", "n": len(valid)}
    matched = [overlap(it["jlens_tokens"], it["explanation"]) for it in valid]
    null = [overlap(a["jlens_tokens"], b["explanation"]) for a in valid for b in valid
            if a["prompt_idx"] != b["prompt_idx"]]
    m, mm = st.mean(matched), st.mean(null)
    sd = st.pstdev(null)
    beats = []
    candidates = []
    for a in valid:
        alternatives = [overlap(a["jlens_tokens"], b["explanation"]) for b in valid
                        if a["prompt_idx"] != b["prompt_idx"]]
        beats.append(overlap(a["jlens_tokens"], a["explanation"]) > max(alternatives))
        candidates.append(len(alternatives))
    return {"status": "exploratory", "n": len(valid), "attempted": len(items),
            "invalid_format": len(items) - len(valid), "matched": m, "mismatched": mm,
            "excess": m - mm, "z": (m - mm) / sd if sd > 1e-3 else None,
            "z_note": "Descriptive effect size using pairwise null SD; not a test statistic",
            "frac_above_own_null": st.mean(beats), "mispaired_candidates": candidates,
            "nominal_rank_chance": st.mean(1 / (n + 1) for n in candidates),
            "rank_chance_note": "Nominal exchangeability reference; ties and prompt clustering matter",
            "jlens_empty_content_count": sum(not norm_words(" ".join(it["jlens_tokens"])) for it in valid),
            "null_pairs": len(null)}


def run(manifest, warmed, run_id):
    import modal
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from jlens import from_hf
    from jlens.hooks import ActivationRecorder
    from rfm_runtime import verbalize

    start = time.monotonic()
    directory = Path("/cache/runs") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    result = {"run_id": run_id, "status": "running", "scope": "phase1_jlens_nla_exploratory",
              "checks": {}, "timings": {}, "gpu": torch.cuda.get_device_name(),
              "protocol": {"prompts": PROMPTS, "positions": POSITIONS, "source_block_index": 42,
                           "base_max_new_tokens": 150, "enable_thinking": False,
                           "jlens_top_k": 12, "nla_max_new_tokens": 256,
                           "metric": "original English word-overlap coefficient",
                           "null": "all ordered pairings from different prompts"},
              "revisions": {k: v.get("revision") for k, v in manifest["repos"].items()},
              "validation": {"independent_nla_reconstruction": "pending",
                             "jlens_provenance": "pending", "sae": "pending"}}
    torch.manual_seed(1729)

    def checkpoint(stage):
        result["stage"] = stage
        result["elapsed_seconds"] = time.monotonic() - start
        result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        modal.Volume.from_name("rfm").commit()
        print(f"rfm: {stage}", flush=True)

    try:
        t = time.monotonic()
        base_path = warmed["snapshots"]["base"]
        tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager",
            local_files_only=True).eval()
        model = from_hf(base, tok, force_bos=False)
        assert model.n_layers == 64 and model.d_model == 5120
        lens_path = Path(warmed["snapshots"]["jlens"]) / (
            "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
        blob = torch.load(lens_path, map_location="cpu", weights_only=True, mmap=True)
        J = blob["J"][42].float().cuda()
        assert J.shape == (5120, 5120) and torch.isfinite(J).all()
        result["timings"]["base_and_lens_load_seconds"] = time.monotonic() - t
        checkpoint("base_loaded")
        samples = []
        generations = []
        t = time.monotonic()
        for pi, prompt in enumerate(PROMPTS):
            chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(chat, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                full = base.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id)
                with ActivationRecorder(model.layers, at=[42]) as recorder:
                    model.forward(full)
                    hs = recorder.activations[42][0].float().cpu()
                for pos in POSITIONS:
                    if pos >= full.shape[1]:
                        continue
                    h = hs[pos]
                    assert torch.isfinite(h).all() and h.norm() > 0
                    logits = model.unembed(J @ h.cuda())
                    token_ids = logits.topk(12).indices.tolist()
                    samples.append({"prompt_idx": pi, "prompt": prompt, "pos": pos,
                        "h": h.clone(), "source": tok.decode(full[0, :pos + 1]),
                        "jlens_token_ids": token_ids,
                        "jlens_tokens": [tok.decode([i]) for i in token_ids]})
            generations.append({"prompt_idx": pi, "prompt": prompt,
                "prompt_tokens": ids.shape[1], "total_tokens": full.shape[1],
                "response": tok.decode(full[0, ids.shape[1]:], skip_special_tokens=True)})
            torch.save(samples, directory / "activations.pt")
            (directory / "generations.json").write_text(json.dumps(generations, indent=2))
            checkpoint(f"source_prompt_{pi + 1}_of_{len(PROMPTS)}")
        result["timings"]["generation_and_extraction_seconds"] = time.monotonic() - t
        result["checks"]["activation_extraction"] = {"passed": True, "samples": len(samples),
            "samples_per_prompt": [sum(s["prompt_idx"] == i for s in samples) for i in range(len(PROMPTS))]}
        del recorder, hs, logits, blob, J, model, base, ids, full
        gc.collect()
        torch.cuda.empty_cache()
        checkpoint("activations_saved")
        verbalize(samples, manifest, warmed, result, checkpoint)
        result["format_status"] = result["status"]
        result["report"] = {"jlens|nla": report(result["readouts"])}
        result["status"] = "exploratory_complete"
        result["remaining"] = ["Independent NLA reconstruction validation",
            "Published Jacobian lens provenance validation", "Compatible SAE with independent feature labels",
            "SAE agreement rows and steering experiment"]
        checkpoint("phase1_collection_complete")
        return result
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        checkpoint("failed")
        raise
