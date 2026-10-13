"""GPU compatibility pilot; produces diagnostics, not study results."""
from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import json
import math
import time
import traceback
from pathlib import Path


TEXTS = [
    "A volcano erupts when pressure from dissolved gases drives magma toward the surface. "
    "Magma contains molten rock, crystals, and gases such as water vapor and carbon dioxide. "
    "As it rises, falling pressure allows bubbles to expand. Explosive eruptions eject ash "
    "and fragments of rock, while quieter eruptions produce lava flows. Scientists monitor "
    "ground movement, small earthquakes, and gas emissions to understand changes beneath "
    "a volcano. Ash can affect aviation and breathing, and volcanic soils can support farming. "
    "The precise hazards depend on the chemistry of the magma and the surrounding landscape.",
    "A compiler translates a program from a source language into machine instructions. "
    "The lexer identifies tokens, and the parser organizes them into a syntax tree. "
    "The type checker verifies that operations use compatible values. An intermediate "
    "representation allows optimization passes to simplify expressions and remove unused "
    "computations. The backend selects instructions for a particular processor and assigns "
    "registers. A linker combines compiled objects with libraries and resolves external "
    "symbols. Debugging information connects the resulting instructions to locations in "
    "the original source code so a developer can inspect program execution.",
]


def run(manifest, warmed, run_id):
    import modal
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from jlens import from_hf
    from jlens.hooks import ActivationRecorder

    start = time.monotonic()
    directory = Path("/cache/runs") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id, "status": "running", "stage": "start",
        "scope": "compatibility_smoke_only", "gpu": torch.cuda.get_device_name(),
        "total_gpu_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "peft", "accelerate", "jlens")},
        "backend": {"attention": "eager", "gdn": "native_pytorch"},
        "revisions": {k: v.get("revision") for k, v in manifest["repos"].items()},
        "checks": {}, "timings": {},
    }
    torch.manual_seed(1729)

    def checkpoint(stage):
        result["stage"] = stage
        result["elapsed_seconds"] = time.monotonic() - start
        result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        modal.Volume.from_name("rfm").commit()
        print(f"rfm: {stage}", flush=True)

    try:
        assert importlib.util.find_spec("fla") is None, "Smoke test requires native PyTorch GDN"
        meta = manifest["repos"]["av"]["metadata"]["nla_meta.yaml"]
        extraction = meta["extraction"]
        layer, d = extraction["layer_index"], extraction["d_model"]
        assert layer == 42 and d == 5120 and extraction["norm"] == "none"
        base_cfg = manifest["repos"]["base"]["metadata"]["config.json"]["text_config"]
        assert base_cfg["hidden_size"] == d and base_cfg["num_hidden_layers"] == 64
        result["checks"]["same_activation_sae"] = {
            "status": "deferred_to_sae_encode", "source_block_index": layer,
            "sae_repo": manifest["repos"]["sae42_2m"]["id"],
            "note": "The encoder separately verifies the pretrained SAE's block and model contract",
        }
        checkpoint("contract_checked")

        t = time.monotonic()
        base_path = warmed["snapshots"]["base"]
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_path, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="eager", local_files_only=True,
        ).eval()
        model = from_hf(base, tokenizer, force_bos=False)
        assert model.n_layers == 64 and model.d_model == d
        result["timings"]["base_load_seconds"] = time.monotonic() - t
        result["checks"]["base_load"] = {"passed": True,
                                              "allocated_gpu_gib": torch.cuda.memory_allocated() / 2**30}
        checkpoint("base_loaded")

        samples = []
        t = time.monotonic()
        for i, text in enumerate(TEXTS):
            ids = model.encode(text, max_length=96)
            assert ids.shape[1] >= 60
            with torch.no_grad(), ActivationRecorder(model.layers, at=[layer]) as recorder:
                model.forward(ids)
                h = recorder.activations[layer][0, -1].float().cpu()
            assert torch.isfinite(h).all() and h.norm() > 0
            samples.append({"prompt_idx": i, "pos": ids.shape[1] - 1,
                            "source": tokenizer.decode(ids[0]), "h": h})
            del recorder
        result["timings"]["activation_extraction_seconds"] = time.monotonic() - t
        torch.save(samples, directory / "activations.pt")
        result["checks"]["activation_extraction"] = {"passed": True,
            "block_index": layer, "samples": len(samples),
            "norms": [s["h"].norm().item() for s in samples]}
        checkpoint("activations_saved")

        t = time.monotonic()
        lens_path = Path(warmed["snapshots"]["jlens"]) / (
            "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
        lens_blob = torch.load(lens_path, map_location="cpu", weights_only=True, mmap=True)
        assert lens_blob["d_model"] == d and lens_blob["n_prompts"] == 1000
        jacobian = lens_blob["J"][layer].float().cuda()
        assert jacobian.shape == (d, d) and torch.isfinite(jacobian).all()
        result["checks"]["published_lens"] = {"passed": True, "n_prompts": 1000,
            "source_block_index": layer, "target_block_index": lens_blob.get("target_layer"),
            "metadata": {k: v for k, v in lens_blob.items()
                         if isinstance(v, (str, int, float, bool))},
        }
        for sample in samples:
            with torch.no_grad():
                transported = jacobian @ sample["h"].cuda()
                logits = model.unembed(transported)
                top = logits.topk(12).indices.tolist()
            sample["jlens_token_ids"] = top
            sample["jlens_tokens"] = [tokenizer.decode([i]) for i in top]
        result["timings"]["published_lens_load_and_readout_seconds"] = time.monotonic() - t
        torch.save(samples, directory / "activations.pt")
        del lens_blob, jacobian, transported, logits
        checkpoint("published_lens_checked")

        # Exactly the reference estimator's first sixteen rows, on one prompt.
        # This checks differentiation and measures cost without fitting a lens.
        t = time.monotonic()
        dim_batch = 8
        from jlens.fitting import SKIP_FIRST_N_POSITIONS
        skip_first = SKIP_FIRST_N_POSITIONS
        ids = model.encode(TEXTS[0], max_length=96).expand(dim_batch, -1)
        with torch.enable_grad(), ActivationRecorder(
                model.layers, at=[layer, 63], start_graph_at=layer) as recorder:
            model.forward(ids)
            source = recorder.activations[layer]
            target = recorder.activations[63]
            forward_seconds = time.monotonic() - t
            cotangent = torch.zeros_like(target)
            chunks, backward_seconds = [], []
            for offset in (0, dim_batch):
                cotangent.zero_()
                for b in range(dim_batch):
                    cotangent[b, skip_first:, offset + b] = 1
                torch.cuda.synchronize()
                tb = time.monotonic()
                grad, = torch.autograd.grad(target, source, cotangent,
                                            retain_graph=(offset == 0))
                rows = grad[:, skip_first:].float().mean(dim=1).detach().cpu()
                torch.cuda.synchronize()
                backward_seconds.append(time.monotonic() - tb)
                chunks.append(rows)
            rows = torch.cat(chunks)
        assert rows.shape == (2 * dim_batch, d)
        assert torch.isfinite(rows).all() and (rows.norm(dim=1) > 0).all()
        torch.save(rows, directory / "jacobian_rows.pt")
        row_seconds = time.monotonic() - t
        result["timings"]["sixteen_jacobian_rows_seconds"] = row_seconds
        result["timings"]["jacobian_forward_seconds"] = forward_seconds
        result["timings"]["jacobian_backward_cold_seconds"] = backward_seconds[0]
        result["timings"]["jacobian_backward_warm_seconds"] = backward_seconds[1]
        result["timings"]["rough_one_prompt_full_jacobian_seconds"] = (
            forward_seconds + backward_seconds[0] + backward_seconds[1] * (math.ceil(d / dim_batch) - 1))
        result["checks"]["jacobian_backward"] = {
            "passed": True, "source_block": layer, "target_block": 63,
            "rows_tested": 2 * dim_batch, "dim_batch": dim_batch, "full_rows": d, "seq_len": ids.shape[1],
            "skip_first": skip_first,
            "row_norms": rows.norm(dim=1).tolist(),
            "estimate_note": "Extrapolated from one warm backward pass at dim_batch=8; tuning batch size changes throughput",
        }
        del recorder, source, target, cotangent, grad, rows, chunks, ids, model, base
        gc.collect()
        torch.cuda.empty_cache()
        result["checks"]["base_freed"] = {"allocated_gpu_gib": torch.cuda.memory_allocated() / 2**30}
        checkpoint("backward_checked")

        return verbalize(samples, manifest, warmed, result, checkpoint)
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        checkpoint("failed")
        raise


def verbalize(samples, manifest, warmed, result, checkpoint, max_new_tokens=256):
    import re
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    meta = manifest["repos"]["av"]["metadata"]["nla_meta.yaml"]
    result["generation_config"] = {"max_new_tokens": max_new_tokens, "do_sample": False}
    t = time.monotonic()
    av_root = Path(warmed["snapshots"]["av"])
    atok = AutoTokenizer.from_pretrained(av_root / "av_base", local_files_only=True)
    av_base = AutoModelForCausalLM.from_pretrained(
        av_root / "av_base", dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager", local_files_only=True,
    ).eval()
    av = PeftModel.from_pretrained(
        av_base, av_root / "av_rl_adapters/iter_000600", is_trainable=False,
    ).eval()
    result["timings"]["av_load_seconds"] = time.monotonic() - t
    result["checks"]["av_load"] = {"passed": True, "step": 600,
        "base": "merged_warmstart", "allocated_gpu_gib": torch.cuda.memory_allocated() / 2**30}
    checkpoint("av_loaded")

    tokens = meta["tokens"]
    marker = tokens["injection_token_id"]
    left, right = tokens["injection_left_neighbor_id"], tokens["injection_right_neighbor_id"]
    assert atok.encode(tokens["injection_char"], add_special_tokens=False) == [marker]
    prompt = meta["prompt_templates"]["actor"].format(injection_char=tokens["injection_char"])
    chat = atok.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pids = atok(chat, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    slots = [i for i in range(1, pids.shape[1] - 1) if pids[0, i].item() == marker
             and pids[0, i - 1].item() == left and pids[0, i + 1].item() == right]
    assert len(slots) == 1, "Injection marker or neighbor tokenization drift"
    slot = slots[0]
    layers = av.get_base_model().model.layers
    state = {"v": None, "count": 0, "relative_norm_error": None}

    def hook(module, args, output):
        resid = output[0] if isinstance(output, tuple) else output
        if resid.shape[1] < 2 or state["v"] is None:
            return output
        assert resid.shape[:2] == pids.shape
        changed = resid.clone()
        h = resid[0, slot].float()
        v = state["v"].to(resid.device).float()
        assert v.norm() > 0 and torch.isfinite(v).all()
        delta = h.norm() * v / v.norm()
        changed[0, slot] = (h + delta).to(resid.dtype)
        state["count"] += 1
        state["relative_norm_error"] = abs((delta.norm() / h.norm()).item() - 1)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = layers[1].register_forward_hook(hook)
    t = time.monotonic()
    readouts = []
    result["readouts"] = readouts
    import re
    try:
        for sample in samples:
            state.update(v=sample["h"], count=0)
            with torch.no_grad():
                generated = av.generate(input_ids=pids, attention_mask=torch.ones_like(pids),
                    max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=atok.eos_token_id)
            text = atok.decode(generated[0, pids.shape[1]:], skip_special_tokens=True)
            match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.S)
            assert state["count"] == 1, "Injection must fire once during prefill"
            assert state["relative_norm_error"] < 1e-5
            readouts.append({"prompt_idx": sample["prompt_idx"], "source": sample["source"],
                "pos": sample.get("pos"),
                "jlens_tokens": sample["jlens_tokens"],
                "text": text, "explanation": match.group(1).strip() if match else None,
                "format_complete": match is not None,
                "generated_tokens": generated.shape[1] - pids.shape[1],
                "token_limit_reached": generated.shape[1] - pids.shape[1] == max_new_tokens,
                "injection_count": state["count"],
                "relative_norm_error": state["relative_norm_error"]})
            checkpoint(f"av_readout_{len(readouts)}_of_{len(samples)}")
        # Negative control: the same marker prompt with injection disabled.
        state.update(v=None, count=0)
        with torch.no_grad():
            generated = av.generate(input_ids=pids, attention_mask=torch.ones_like(pids),
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=atok.eos_token_id)
        no_injection = atok.decode(generated[0, pids.shape[1]:], skip_special_tokens=True)
        assert state["count"] == 0
    finally:
        handle.remove()
    result["timings"]["av_generations_seconds"] = time.monotonic() - t
    result["timings"]["av_generation_count"] = len(samples) + 1
    result["readouts"] = readouts
    result["no_injection_control"] = no_injection
    result["checks"]["nla_injection"] = {"passed": True, "injection_block_index": 1,
        "method": "h + norm(h) * v / norm(v)", "samples": len(readouts),
        "format_complete_count": sum(r["format_complete"] for r in readouts)}
    result["status"] = "passed" if all(r["format_complete"] for r in readouts) else "format_incomplete"
    result["remaining"] = ["Validate published Jacobian lens provenance and readout performance",
        "Validate NLA reconstruction on independent held-out data",
        "Obtain or train block-42 SAE and independently validate feature labels",
        "Run prompt-level mismatch controls and causal steering experiments"]
    checkpoint("smoke_complete")
    return result


def resume_nla(manifest, warmed, run_id, previous_run_id):
    import modal
    import torch
    start = time.monotonic()
    directory = Path("/cache/runs") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    previous = Path("/cache/runs") / previous_run_id
    prior = json.loads((previous / "result.json").read_text())
    samples = torch.load(previous / "activations.pt", map_location="cpu", weights_only=True)
    result = {"run_id": run_id, "status": "running", "scope": "nla_resume_only",
              "inherited_from": previous_run_id, "gpu": torch.cuda.get_device_name(),
              "checks": {}, "timings": {}, "prior_checks": prior["checks"]}
    torch.manual_seed(1729)

    def checkpoint(stage):
        result["stage"] = stage
        result["elapsed_seconds"] = time.monotonic() - start
        result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        modal.Volume.from_name("rfm").commit()
        print(f"rfm: {stage}", flush=True)

    try:
        assert importlib.util.find_spec("fla") is None
        return verbalize(samples, manifest, warmed, result, checkpoint)
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        checkpoint("failed")
        raise
