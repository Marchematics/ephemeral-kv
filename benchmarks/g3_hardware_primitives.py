#!/usr/bin/env python
"""G3 hardware primitives for the session-mobility law.

Unlike the accounting scripts, this runner measures the machine it is executed on.
It deliberately separates three costs:

1. one-way CPU->GPU movement of a history-sized KV byte payload when it fits;
2. frozen-model prefill of full-history and active-view token counts;
3. the composed cold-route tax used by the mobility gate.

A row is marked measured only when the corresponding operation actually ran. OOM or
unsupported long contexts are recorded as infeasible, never replaced by extrapolation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median


def percentile(xs, q):
    if not xs:
        return None
    ys=sorted(xs)
    pos=(len(ys)-1)*q
    lo=int(pos)
    hi=min(len(ys)-1, lo+1)
    if lo==hi:
        return ys[lo]
    return ys[lo]*(hi-pos)+ys[hi]*(pos-lo)


def kv_payload_bytes(tokens: int, kv_bytes_per_token: int) -> int:
    return int(tokens) * int(kv_bytes_per_token)


def summarize_times(xs):
    return {
        "n": len(xs),
        "p50_s": median(xs) if xs else None,
        "p95_s": percentile(xs, 0.95),
        "min_s": min(xs) if xs else None,
    }


def measure_h2d(payload_bytes: int, *, repeats: int, device: str):
    import torch

    if not device.startswith("cuda"):
        return {"status": "unsupported", "reason": "H2D measurement requires CUDA"}

    free, total = torch.cuda.mem_get_info(device)
    # Leave room for the model/runtime; this benchmark is a mobility primitive, not an
    # invitation to trigger allocator OOM.
    if payload_bytes > int(free * 0.70):
        return {
            "status": "infeasible",
            "reason": "payload exceeds 70% of currently free GPU memory",
            "payload_bytes": payload_bytes,
            "free_gpu_bytes": int(free),
        }

    try:
        cpu = torch.empty(payload_bytes, dtype=torch.uint8, pin_memory=True)
        gpu = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
        torch.cuda.synchronize()
        times=[]
        for _ in range(repeats):
            t0=time.perf_counter()
            gpu.copy_(cpu, non_blocking=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-t0)
        del gpu, cpu
        torch.cuda.empty_cache()
        return {
            "status": "measured",
            "payload_bytes": payload_bytes,
            "times": summarize_times(times),
            "effective_GBps_p50": (
                payload_bytes / summarize_times(times)["p50_s"] / 1e9
                if times else None
            ),
        }
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return {"status": "infeasible", "reason": str(e), "payload_bytes": payload_bytes}


def load_model(name: str, device: str, dtype_name: str, attn_implementation: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype={
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }[dtype_name]
    tok=AutoTokenizer.from_pretrained(name, use_fast=True)
    kwargs={"torch_dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"]=attn_implementation
    model=AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(device)
    model.eval()
    return model, tok


def measure_prefill(model, *, tokens: int, repeats: int, device: str):
    import torch

    # Deterministic valid IDs; no dataset download is needed for the systems primitive.
    vocab=int(model.config.vocab_size)
    input_ids=(torch.arange(tokens, device=device, dtype=torch.long) % max(2, vocab-1)).unsqueeze(0)

    times=[]
    peaks=[]
    try:
        for _ in range(repeats):
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
            t0=time.perf_counter()
            with torch.inference_mode():
                # logits_to_keep avoids materializing history-sized vocabulary logits on
                # recent Transformers; fall back for models that do not expose it.
                try:
                    model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
                except TypeError:
                    model(input_ids=input_ids, use_cache=True)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                peaks.append(int(torch.cuda.max_memory_allocated(device)))
            times.append(time.perf_counter()-t0)
        return {
            "status":"measured",
            "tokens":tokens,
            "times":summarize_times(times),
            "tokens_per_s_p50":tokens/summarize_times(times)["p50_s"],
            "peak_gpu_bytes_p50":median(peaks) if peaks else None,
        }
    except RuntimeError as e:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {"status":"infeasible","tokens":tokens,"reason":str(e)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--histories", type=int, nargs="+", default=[32768,131072,524288,1048576])
    p.add_argument("--active", type=int, nargs="+", default=[2048,4096,8192,16384])
    p.add_argument("--kv-bytes-per-token", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["auto","float16","bfloat16","float32"], default="bfloat16")
    p.add_argument("--attn-implementation", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-full-prefill", action="store_true")
    args=p.parse_args(argv)

    import torch
    model, _ = load_model(args.model,args.device,args.dtype,args.attn_implementation)

    rows=[]
    for h in args.histories:
        transfer=measure_h2d(
            kv_payload_bytes(h,args.kv_bytes_per_token),
            repeats=args.repeats,
            device=args.device,
        )
        full_prefill=(
            {"status":"skipped"}
            if args.skip_full_prefill
            else measure_prefill(model,tokens=h,repeats=args.repeats,device=args.device)
        )
        rows.append({
            "history_tokens":h,
            "kv_payload_bytes":kv_payload_bytes(h,args.kv_bytes_per_token),
            "full_kv_h2d":transfer,
            "full_reprefill":full_prefill,
        })

    active_rows=[
        measure_prefill(model,tokens=w,repeats=args.repeats,device=args.device)
        for w in args.active
    ]

    payload={
        "schema":"ephemeral-kv-g3-hardware-primitives-v1",
        "kind":"hardware_measurement",
        "model":args.model,
        "device":str(torch.cuda.get_device_name(args.device)) if args.device.startswith("cuda") else args.device,
        "config":vars(args),
        "history_rows":rows,
        "active_prefill_rows":active_rows,
        "honesty":(
            "H2D and prefill are directly measured primitives. Indexed lookup must be "
            "measured on the real trace separately before composing end-to-end mobility tax. "
            "Infeasible/OOM rows are retained as such and are never extrapolated."
        ),
    }
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
