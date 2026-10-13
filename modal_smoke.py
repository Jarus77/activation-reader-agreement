"""Bounded Modal preflight and GPU smoke test. Remote names are deliberately rfm."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
REPOS = {
    "base": {"id": "Qwen/Qwen3.6-27B", "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"},
    "av": {"id": "ceselder/qwen3.6-27b-nla-rl", "revision": "5a13b7ec21a69fcdd0fb24d5edbe96a92aef4b9f"},
    "jlens": {"id": "neuronpedia/jacobian-lens", "revision": "9fa267f841d62aa468dab5b9b38c3dc45a82d863"},
    "sae42_2m": {"id": "ceselder/qwen36-27b-sae2m-l42", "revision": "3233a3d8744da53999fe5e6d36a8357b873330ba"},
    "label": {"id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
}


def metadata(repos):
    import json
    from huggingface_hub import HfApi, hf_hub_download
    import yaml

    api = HfApi()
    result = {"repos": {}}
    for role, checkpoint in repos.items():
        repo, revision = checkpoint["id"], checkpoint["revision"]
        info = api.model_info(repo, revision=revision, files_metadata=True)
        assert info.sha == revision, "Checkpoint revision does not match the pinned experiment"
        files = [{"path": f.rfilename, "bytes": f.size} for f in info.siblings]
        result["repos"][role] = {"id": repo, "revision": info.sha, "files": files}
        names = {f["path"] for f in files}
        wanted = {
            "base": ["config.json"],
            "av": ["nla_meta.yaml", "av_base/config.json", "run_config.yaml",
                   "av_rl_adapters/iter_000600/adapter_config.json"],
            "jlens": [],
            "sae42_2m": ["saes_Qwen_Qwen3.6-27B_batch_top_k/resid_post_layer_42/trainer_0/config.json", "verify.json"],
            "label": ["config.json"],
        }[role]
        for name in wanted:
            if name in names:
                p = hf_hub_download(repo, name, revision=info.sha)
                with open(p) as stream:
                    data = yaml.safe_load(stream) if name.endswith(".yaml") else json.load(stream)
                result["repos"][role].setdefault("metadata", {})[name] = data
        print(f"rfm: metadata {role} ready", flush=True)
    result["status"] = "metadata_only"
    result["gpu_used"] = False
    return result


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def write_agreement_report(result):
    sae_path = OUT / "sae_comparisons.json"
    if sae_path.exists():
        sae = json.loads(sae_path.read_text())
        if sae.get("agreement_run_id") == result.get("run_id"):
            from analyze_sae import write_report
            write_report(sae)
            return
    row = result.get("report", {}).get("jlens|nla", {})
    if "matched" not in row:
        return
    z = "n/a" if row["z"] is None else f"{row['z']:+.2f}"
    body = (
        "# Qwen3.6-27B exploratory agreement\n\n"
        "Instrument validation remains pending; these are measured collection results.\n\n"
        "| Pair | Matched | Mismatched | Excess | z (descriptive) | Beats all mispairings |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"| J-lens vs NLA | {row['matched']:.3f} | {row['mismatched']:.3f} | "
        f"{row['excess']:+.3f} | {z} | {row['frac_above_own_null']:.2f} |\n"
        "| NLA vs SAE | pending | pending | pending | pending | pending |\n"
        "| J-lens vs SAE | pending | pending | pending | pending | pending |\n\n"
        f"Valid descriptions: {row['n']}/{row['attempted']}; invalid format: {row['invalid_format']}. "
        f"Nominal rank reference: {row['nominal_rank_chance']:.3f}. "
        "Positions from the same prompt are dependent; the displayed z is not a significance test.\n\n"
        "The SAE rows and steering table require a compatible SAE and independent feature labels. "
        "NLA reconstruction and published lens provenance checks also remain pending.\n"
    )
    (OUT / "agreement_report.md").write_text(body)


def warm(manifest):
    import os
    from huggingface_hub import snapshot_download
    from pathlib import Path
    import json
    import time
    import modal

    started = time.monotonic()
    result = {"status": "warming", "snapshots": {}, "revisions": {}}
    selected = {
        "base": ["*.json", "*.safetensors", "tokenizer*", "vocab.json", "merges.txt"],
        "av": ["av_base/*", "av_rl_adapters/iter_000600/*", "nla_meta.yaml",
               "data/example_activations.parquet"],
        "jlens": ["qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"],
    }
    try:
        for role, patterns in selected.items():
            r = manifest["repos"][role]
            path = snapshot_download(r["id"], revision=r["revision"],
                                     allow_patterns=patterns, max_workers=4,
                                     cache_dir="/cache/hf")
            result["snapshots"][role] = path
            result["revisions"][role] = r["revision"]
            print(f"rfm: cache {role} ready", flush=True)
        result.update(status="ready", elapsed_seconds=time.monotonic() - started,
                      gpu_used=False)
        Path("/cache/warm.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        modal.Volume.from_name("rfm").commit()


def gpu_worker(manifest, warmed, run_id):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_runtime import run
    return run(manifest, warmed, run_id)


def nla_worker(manifest, warmed, run_id, previous_run_id):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_runtime import resume_nla
    return resume_nla(manifest, warmed, run_id, previous_run_id)


def agreement_worker(manifest, warmed, run_id):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_agreement import run
    return run(manifest, warmed, run_id)


def sae_warm_worker(manifest):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_sae import warm
    return warm(manifest)


def sae_encode_worker(manifest, warmed, run_id, agreement_run_id):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_sae import encode
    return encode(manifest, warmed, run_id, agreement_run_id)


def sae_label_worker(warmed, run_id, encoding_run_id):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_sae import annotate
    return annotate(warmed, run_id, encoding_run_id)


def sae_review_worker(base_warmed, manifest, run_id, annotation_run_id):
    import sys
    sys.path.insert(0, "/opt")
    from rfm_sae import review
    return review(base_warmed, manifest, run_id, annotation_run_id)


def contract_worker(manifest, warmed):
    import torch
    from pathlib import Path
    from transformers import Qwen3_5ForConditionalGeneration, AutoModelForCausalLM
    import json
    import modal

    path = Path(warmed["snapshots"]["jlens"]) / (
        "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
    blob = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    scalars = {k: v for k, v in blob.items() if isinstance(v, (str, int, float, bool))}
    result = {"status": "ready", "lens_metadata": scalars,
              "lens_keys": list(blob), "J42_shape": list(blob["J"][42].shape),
              "gpu_used": False}
    assert blob["d_model"] == 5120 and blob["n_prompts"] == 1000
    assert blob["J"][42].shape == (5120, 5120)
    return result


def gpu_image():
    return (modal.Image.debian_slim(python_version="3.11").apt_install("git")
            .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
            .pip_install("transformers==5.5.4", "peft==0.19.1", "accelerate==1.12.0",
                         "safetensors==0.8.0", "pyyaml==6.0.3", "pyarrow==26.0.0")
            .pip_install("git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e")
            .env({"HF_HOME": "/cache/hf", "HF_HUB_OFFLINE": "1",
                  "HF_HUB_DISABLE_PROGRESS_BARS": "1", "WANDB_MODE": "disabled",
                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            .add_local_file(ROOT / "rfm_runtime.py", "/opt/rfm_runtime.py")
            .add_local_file(ROOT / "rfm_agreement.py", "/opt/rfm_agreement.py")
            .add_local_file(ROOT / "rfm_sae.py", "/opt/rfm_sae.py"))


def launch(stage):
    run_id = uuid.uuid4().hex[:12]
    app = modal.App("rfm", tags={"rfm": "rfm"})
    image = modal.Image.debian_slim(python_version="3.11")
    kwargs = dict(name="rfm", retries=0, max_containers=1, scaledown_window=2)
    if stage == "metadata":
        image = image.pip_install("huggingface_hub==0.36.0", "pyyaml==6.0.3")
        fn = app.function(image=image, timeout=300, **kwargs)(metadata)
        inputs = (REPOS,)
    elif stage in ("warm", "sae_warm"):
        manifest = json.loads((OUT / "metadata.json").read_text())
        image = image.pip_install("huggingface_hub==0.36.0", "pyyaml==6.0.3").env({
            "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HOME": "/cache/hf",
        })
        if stage == "sae_warm":
            image = image.add_local_file(ROOT / "rfm_sae.py", "/opt/rfm_sae.py")
        fn = app.function(image=image, timeout=1800, cpu=4, memory=8192,
                          volumes={"/cache": modal.Volume.from_name("rfm", create_if_missing=True)},
                          **kwargs)(sae_warm_worker if stage == "sae_warm" else warm)
        inputs = (manifest,)
    elif stage in ("sae_encode", "sae_labels", "sae_review"):
        manifest = json.loads((OUT / "metadata.json").read_text())
        warmed = json.loads((OUT / "sae_warm.json").read_text())
        if warmed["status"] != "ready":
            raise ValueError("SAE and annotation weights must be cached before GPU allocation")
        if stage == "sae_encode":
            agreement = json.loads((OUT / "agreement.json").read_text())
            if agreement["status"] != "exploratory_complete":
                raise ValueError("SAE comparison requires completed agreement collection")
            worker = sae_encode_worker
            inputs = (manifest, warmed, run_id, agreement["run_id"])
        elif stage == "sae_labels":
            encoded = json.loads((OUT / "sae_encode.json").read_text())
            if encoded["status"] != "encoding_complete":
                raise ValueError("Annotation requires successful SAE encoding")
            worker = sae_label_worker
            inputs = (warmed, run_id, encoded["run_id"])
        else:
            annotations = json.loads((OUT / "sae_labels.json").read_text())
            if annotations["status"] != "annotation_complete":
                raise ValueError("Independent review requires frozen completed labels")
            base_warmed = json.loads((OUT / "warm.json").read_text())
            assert base_warmed["revisions"]["base"] == manifest["repos"]["base"]["revision"]
            worker = sae_review_worker
            inputs = (base_warmed, manifest, run_id, annotations["run_id"])
        image = gpu_image()
        fn = app.function(image=image, gpu="H200" if stage != "sae_labels" else ["L4", "A10", "L40S", "H200"],
                          timeout=1200 if stage != "sae_labels" else 1800, cpu=4,
                          memory=98304 if stage != "sae_labels" else 16384,
                          volumes={"/cache": modal.Volume.from_name("rfm")}, **kwargs)(worker)
    elif stage in ("gpu", "nla", "agreement", "build", "contract"):
        manifest = json.loads((OUT / "metadata.json").read_text())
        if stage in ("gpu", "nla", "agreement", "contract"):
            warmed = json.loads((OUT / "warm.json").read_text())
            if warmed["status"] != "ready":
                raise ValueError("Weights must be cached before allocating a GPU")
            for role in ("base", "av", "jlens"):
                if manifest["repos"][role]["revision"] != warmed["revisions"][role]:
                    raise ValueError("Metadata revisions changed after caching")
            inputs = (manifest, warmed, run_id) if stage in ("gpu", "agreement") else (manifest, warmed)
            if stage == "nla":
                previous = json.loads((OUT / "gpu.json").read_text())
                if not previous["checks"].get("jacobian_backward", {}).get("passed"):
                    raise ValueError("Resume requires a successful activation and differentiation check")
                inputs = (manifest, warmed, run_id, previous["run_id"])
            if stage == "agreement":
                pilot = json.loads((OUT / "nla.json").read_text())
                if pilot["status"] != "passed":
                    raise ValueError("Agreement collection requires complete NLA pilot descriptions")
        image = gpu_image()
        if stage == "contract":
            fn = app.function(image=image, timeout=300, cpu=2, memory=8192,
                              volumes={"/cache": modal.Volume.from_name("rfm")}, **kwargs)(contract_worker)
        else:
            worker = {"nla": nla_worker, "agreement": agreement_worker}.get(stage, gpu_worker)
            fn = app.function(image=image, gpu="H200", timeout=5400 if stage == "agreement" else 1800,
                              cpu=8, memory=131072,
                              volumes={"/cache": modal.Volume.from_name("rfm")}, **kwargs)(
                                  worker)
    else:
        raise ValueError(stage)
    OUT.mkdir(exist_ok=True)
    record = {"stage": stage, "name": "rfm", "run_id": run_id,
              "created_unix": time.time()}
    with modal.enable_output():
        with app.run(detach=True):
            if stage == "build":
                save_json(OUT / "build.json", {"status": "ready", "app_id": app.app_id})
                print("rfm: GPU image ready; no GPU allocated", flush=True)
                return
            call = fn.spawn(*inputs)
            record.update(app_id=app.app_id, call_id=call.object_id)
            save_json(OUT / f"{stage}_launch.json", record)
            save_json(OUT / f"{stage}_{run_id}_launch.json", record)
            print(f"rfm submitted: {app.app_id}; call {call.object_id}", flush=True)
        try:
            result = call.get(timeout=6000 if stage == "agreement" else 2400)
        except Exception:
            if stage in ("gpu", "nla", "agreement", "sae_encode", "sae_labels", "sae_review"):
                try:
                    remote_path = f"/runs/{run_id}/result.json"
                    raw = b"".join(modal.Volume.from_name("rfm").read_file(remote_path))
                    diagnostic = json.loads(raw)
                    save_json(OUT / f"{stage}.json", diagnostic)
                    save_json(OUT / f"{stage}_{run_id}.json", diagnostic)
                except Exception as retrieval_error:
                    print(f"rfm: diagnostic retrieval failed: {type(retrieval_error).__name__}")
            raise
    if stage == "agreement" and (OUT / "agreement.json").exists():
        previous = json.loads((OUT / "agreement.json").read_text())
        if previous.get("run_id") != result.get("run_id"):
            # SAE analyses are tied to exact activation IDs; retain their previous results.
            for name in ("sae_encode.json", "sae_labels.json", "sae_review.json", "sae_comparisons.json",
                         "prompt_overlap.json", "sae_comparisons_report.md", "prompt_overlap_report.md"):
                path = OUT / name
                if path.exists():
                    backup = OUT / "archive" / str(previous.get("run_id", "previous")) / name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(backup)
    save_json(OUT / f"{stage}.json", result)
    save_json(OUT / f"{stage}_{run_id}.json", result)
    if stage == "agreement":
        write_agreement_report(result)
    elif stage in ("sae_labels", "sae_review") and result.get("status") == "annotation_complete":
        from analyze_sae import main as analyze
        analyze()
    print(f"rfm: saved {stage} result locally", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["metadata", "warm", "sae_warm", "sae_encode", "sae_labels", "sae_review", "gpu", "nla", "agreement", "build", "contract", "status", "progress", "retrieve", "logs", "cancel"])
    parser.add_argument("target", nargs="?", choices=["metadata", "warm", "sae_warm", "sae_encode", "sae_labels", "sae_review", "gpu", "nla", "agreement", "contract"])
    args = parser.parse_args()
    if args.stage in ("status", "progress", "retrieve", "logs", "cancel"):
        if args.target is None:
            parser.error("status/retrieve requires a target")
        record = json.loads((OUT / f"{args.target}_launch.json").read_text())
        if args.stage == "cancel":
            call = modal.FunctionCall.from_id(record["call_id"])
            call.cancel(terminate_containers=True)
            record["cancelled"] = True
            save_json(OUT / f"{args.target}_{record['run_id']}_launch.json", record)
            print("rfm: target call cancelled")
            return
        if args.stage == "progress":
            remote_path = f"/runs/{record['run_id']}/result.json"
            try:
                raw = b"".join(modal.Volume.from_name("rfm").read_file(remote_path))
            except FileNotFoundError:
                print(json.dumps({"run_id": record["run_id"], "stage": "awaiting_first_checkpoint"}))
                return
            value = json.loads(raw)
            save_json(OUT / f"{args.target}_progress.json", value)
            summary = {k: value.get(k) for k in
                       ("run_id", "status", "stage", "gpu", "elapsed_seconds", "timings", "checks")}
            summary["readouts_completed"] = len(value.get("readouts", []))
            summary["formats_complete"] = sum(r.get("format_complete", False)
                                               for r in value.get("readouts", []))
            if "labels" in value:
                summary["labels_proposed"] = len(value["labels"])
                summary["labels_checked"] = sum("validation_text" in r for r in value["labels"])
                summary["labels_accepted"] = sum(r.get("accepted", False) for r in value["labels"])
                if value.get("review_model"):
                    summary["independent_review_checked"] = sum("review_text" in r for r in value["labels"])
                    summary["independent_review_candidates"] = value["review_candidates"]
            print(json.dumps(summary, indent=2))
            return
        if args.stage == "logs":
            import subprocess
            import sys
            try:
                subprocess.run([sys.executable, "-m", "modal", "app", "logs", record["app_id"]],
                               timeout=15, check=True)
            except subprocess.TimeoutExpired:
                pass
            return
        call = modal.FunctionCall.from_id(record["call_id"])
        if args.stage == "status":
            graph = call.get_call_graph()
            print(json.dumps([{ "function": node.function_name, "status": node.status.name,
                                "task_assigned": bool(node.task_id)}
                              for node in graph], indent=2))
        else:
            value = call.get(timeout=0)
            save_json(OUT / f"{args.target}.json", value)
            if args.target == "agreement":
                write_agreement_report(value)
            elif args.target in ("sae_labels", "sae_review") and value.get("status") == "annotation_complete":
                from analyze_sae import main as analyze
                analyze()
            print(f"rfm: saved {args.target} result locally")
    else:
        launch(args.stage)


if __name__ == "__main__":
    main()
