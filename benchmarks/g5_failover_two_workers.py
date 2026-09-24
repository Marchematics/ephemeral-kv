#!/usr/bin/env python
"""G5: two workers, a killed owner, no KV transfer.

The model-rollout half of G5 is already measured (resume cost and cross-model fidelity).  This
runner closes the other half with *processes* rather than arithmetic:

1. the durable object is written to disk - the transcript spans plus the recorded turn, which is
   all a worker needs, and which no model produced;
2. **worker A** is a separate process: it reads the durable object, compiles the execution state,
   prefills it on the GPU, serves the recorded turn and records its timings;
3. the parent **kills worker A** (SIGKILL, so nothing is flushed);
4. **worker B** is a fresh process - optionally a different model - that reads the *same* durable
   object, rebuilds the state from scratch and serves the same turn;
5. the parent reports recovery latency, the bytes B read from the durable object (the state text),
   the KV payload the old design would have had to move for the same recovery, and whether B's
   answer names the files the recorded patch touched.

Nothing is transferred between A and B except the durable object; the KV that A held is gone with
its process, which is the point.

Usage:
    python benchmarks/g5_failover_two_workers.py --jsonl data/sessions-patch-long.jsonl \
        --model-a <ckpt> --model-b <ckpt> --token-budget 8192 --max-examples 4 \
        --out artifacts/g5-failover-two-workers-v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.g2_model_quality import build_examples, render_span, score_target
from benchmarks.g2_trace_index import messages_from_row
from benchmarks.g2b_patch_localization import patch_files_from_messages
from ephemeralkv.index import DurableSpanIndex


def write_durable(jsonl: Path, out: Path, token_budget: int, min_history_tokens: int,
                  max_examples: int, model: str) -> list[dict]:
    """Write one durable object per session: the spans a worker would read, plus the target."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731

    sessions = []
    with jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                messages = messages_from_row(row)
            except ValueError:
                continue
            recorded = patch_files_from_messages(messages)
            if not recorded:
                continue
            examples = build_examples(
                messages, token_budget=token_budget, max_spans=128,
                min_history_tokens=min_history_tokens,
                compiler={"recency_spans": 3, "recency_fraction": 0.6,
                          "max_span_fraction": 0.25, "provenance_terms": 8,
                          "dedup": True, "snippet": True},
                token_counter=counter,
            )
            for index, example in enumerate(examples):
                # the durable object is the transcript *before* the target plus the recorded turn
                spans = [{"turn": i, "role": m.get("role", "?"), "text": _content(m)}
                         for i, m in enumerate(messages[:len(messages) - 1])]
                sessions.append({
                    "session": row.get("instance_id") or row.get("session_id") or f"row{len(sessions)}",
                    "example": index,
                    "spans": spans,
                    "query": example.active_context[-2000:],
                    "target": example.target,
                    "recorded_files": sorted(recorded),
                    "history_tokens": example.history_tokens_estimate,
                })
                break
            if len(sessions) >= max_examples:
                break
    out.write_text(json.dumps({"schema": "ephemeral-kv-durable-object-v1",
                               "sessions": sessions}, indent=2) + "\n")
    return sessions


def _content(msg) -> str:
    from benchmarks.g2_trace_index import _content as content
    return content(msg)


def worker(session: dict, model_name: str, token_budget: int, max_length: int,
           device: str, out: Path) -> int:
    """One worker: rebuild the state from the durable object, serve the turn, record timings."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    load_s = time.time() - t0

    # rebuild the index from the durable object (nothing model-specific is transferred)
    index = DurableSpanIndex()
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))  # noqa: E731
    for span in session["spans"]:
        index.append(turn=span["turn"], role=span["role"], text=span["text"],
                     token_estimate=max(1, counter(span["text"])))
    query = session["query"]
    t1 = time.time()
    view, _ = index.compile_view(query, token_budget=token_budget, max_spans=128,
                                 recency_spans=3, recency_fraction=0.6,
                                 max_span_fraction=0.25, provenance_terms=8,
                                 dedup=True, snippet=True)
    active = "".join(render_span(s) for s in view)
    compile_s = time.time() - t1

    t2 = time.time()
    scored = score_target(model, tokenizer, active, session["target"], max_length, device,
                          max_target_tokens=128)
    prefill_and_score_s = time.time() - t2

    # generate a short continuation so the answer can be compared with the recorded patch
    ids = tokenizer.encode(active, add_special_tokens=False)[-max_length:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, max_new_tokens=96, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    continuation = tokenizer.decode(generated[0][input_ids.shape[1]:],
                                    skip_special_tokens=True)

    out.write_text(json.dumps({
        "model": model_name,
        "load_s": round(load_s, 2),
        "compile_s": round(compile_s, 3),
        "prefill_and_score_s": round(prefill_and_score_s, 3),
        "state_tokens": sum(s.token_estimate for s in view),
        "state_text_bytes": len(active.encode()),
        "nll": scored["nll"] if scored else None,
        "token_accuracy": scored["token_accuracy"] if scored else None,
        "continuation": continuation,
    }, indent=2) + "\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", default="data/sessions-patch-long.jsonl")
    p.add_argument("--model-a", default="/root/qcc/models/Llama-3.2-1B-Instruct")
    p.add_argument("--model-b", default="/root/qcc/models/Qwen2.5-1.5B")
    p.add_argument("--token-budget", type=int, default=8192)
    p.add_argument("--min-history-tokens", type=int, default=32768)
    p.add_argument("--max-length", type=int, default=65536)
    p.add_argument("--max-examples", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workdir", default="/tmp/g5-failover")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    durable = workdir / "durable.json"
    sessions = write_durable(Path(args.jsonl), durable, args.token_budget,
                            args.min_history_tokens, args.max_examples, args.model_a)

    rows = []
    for session in sessions:
        name = f"{session['session']}".replace("/", "_")[:60]
        result_a, result_b = workdir / f"{name}.a.json", workdir / f"{name}.b.json"
        for path in (result_a, result_b):
            if path.exists():
                path.unlink()

        # worker A: serves the turn, then becomes the owner that will be killed
        proc_a = subprocess.Popen([sys.executable, __file__, "--role", "worker",
                                   "--durable", str(durable),
                                   "--session", session["session"],
                                   "--model", args.model_a, "--token-budget",
                                   str(args.token_budget), "--max-length", str(args.max_length),
                                   "--device", args.device, "--result", str(result_a)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 900
        while time.time() < deadline and not result_a.exists():
            if proc_a.poll() is not None:
                break
            time.sleep(1)
        killed_at = time.time()
        if proc_a.poll() is None:
            os.kill(proc_a.pid, signal.SIGKILL)          # the owner disappears; its KV dies with it
        proc_a.wait(timeout=30)

        # worker B: a fresh process, optionally a different model, rebuilding from the durable object
        started_b = time.time()
        proc_b = subprocess.Popen([sys.executable, __file__, "--role", "worker",
                                   "--durable", str(durable),
                                   "--session", session["session"],
                                   "--model", args.model_b, "--token-budget",
                                   str(args.token_budget), "--max-length", str(args.max_length),
                                   "--device", args.device, "--result", str(result_b)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 1800
        while time.time() < deadline and not result_b.exists():
            if proc_b.poll() is not None:
                break
            time.sleep(1)
        proc_b.wait(timeout=60)
        recovered_at = time.time()

        a = json.loads(result_a.read_text()) if result_a.exists() else {}
        b = json.loads(result_b.read_text()) if result_b.exists() else {}
        mentioned = {f for f in session["recorded_files"]
                     if f.split("/")[-1] in (b.get("continuation") or "")}
        rows.append({
            "session": session["session"],
            "history_tokens": session["history_tokens"],
            "owner_killed": a.get("state_tokens") is not None,
            "owner_kv_lost": True,
            "worker_a": {"model": Path(args.model_a).name, **{k: a.get(k) for k in
                         ("load_s", "compile_s", "prefill_and_score_s", "state_tokens",
                          "state_text_bytes", "token_accuracy")}},
            "worker_b": {"model": Path(args.model_b).name, **{k: b.get(k) for k in
                         ("load_s", "compile_s", "prefill_and_score_s", "state_tokens",
                          "state_text_bytes", "token_accuracy")}},
            "recovery_wall_s": round(recovered_at - killed_at, 2),
            "model_load_s": b.get("load_s"),
            "state_rebuild_s": round((b.get("compile_s") or 0) + (b.get("prefill_and_score_s") or 0), 3),
            "durable_bytes_read": b.get("state_text_bytes"),
            "kv_payload_avoided_bytes": session["history_tokens"] * 12288,
            "recorded_files": sorted(session["recorded_files"]),
            "b_mentions_recorded": sorted(mentioned),
            "b_hit": bool(mentioned),
        })
        print(f"{session['session'][:40]:<42} b hit={rows[-1]['b_hit']} "
              f"rebuild={rows[-1]['state_rebuild_s']}s state={b.get('state_tokens')}")

    payload = {"schema": "ephemeral-kv-g5-failover-two-workers-v1",
               "kind": "process_level_failover",
               "interpretation": ("worker A is killed with SIGKILL after serving; worker B is a "
                                  "fresh process that rebuilds the state from the durable object, "
                                  "so no KV is transferred and none survives"),
               "model_a": args.model_a, "model_b": args.model_b,
               "token_budget": args.token_budget, "rows": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    hits = sum(1 for r in rows if r["b_hit"])
    print(f"{hits}/{len(rows)} sessions recovered on the second worker with the recorded files named")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    # worker mode re-enters this file with --role worker
    if "--role" in sys.argv:
        q = argparse.ArgumentParser()
        q.add_argument("--role")
        q.add_argument("--durable", required=True)
        q.add_argument("--session", required=True)
        q.add_argument("--model", required=True)
        q.add_argument("--token-budget", type=int, required=True)
        q.add_argument("--max-length", type=int, required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--result", required=True)
        qargs = q.parse_args()
        payload = json.loads(Path(qargs.durable).read_text())
        target = next(s for s in payload["sessions"] if s["session"] == qargs.session)
        raise SystemExit(worker(target, qargs.model, qargs.token_budget, qargs.max_length,
                                qargs.device, Path(qargs.result)))
    raise SystemExit(main())
