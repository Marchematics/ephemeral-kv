#!/bin/bash
# The killer table with the composed buckets folded in: does the state that must move grow with
# the session, out to a million tokens?
#
# The real corpus stops at ~156K tokens, so the table's past-that rows were composed primitives.
# `build_composed_long_sessions.py` builds real transcripts into 145K/278K/545K/1.07M-token
# histories (153/302/512/1321 turns), and the arms below measure them: the shipped 8,192-token
# state (fidelity and decision), a 131,072-token reference view - the most a resident system can
# hold on this card - and the state's tokens and bytes, which are CPU measurements.
#
# Read the output rows as: raw history, state tokens, state bytes, fidelity against the reference,
# and whether the decision holds.  The claim the table tests is that the first column grows by ~7x
# across these buckets while the middle columns do not move.
set -x
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
while pgrep -f "^/root/qcc/venv/bin/python benchmarks/g2_model_quality.py|^/root/qcc/venv/bin/python benchmarks/g2b_patch_localization.py" > /dev/null; do sleep 20; done

fidelity=(--fidelity artifacts/g2-compiler-tailstate-tf0.6-b8192-v3.json)
end_task=(--end-task artifacts/g2b-patch-localization-tailstate-tf0.6-b8192-v3.json)
for bucket in 128k 256k 512k 1m; do
  fidelity+=(--fidelity "artifacts/g2-composed-${bucket}-state8192-v1.json")
  end_task+=(--end-task "artifacts/g2b-patch-localization-composed-${bucket}-b8192-v1.json")
done

/root/qcc/venv/bin/python benchmarks/g2_killer_table.py \
  "${fidelity[@]}" "${end_task[@]}" \
  --out artifacts/g2-killer-table-composed-v1.json

# the state's size in the units a cold route moves, at each composed length
/root/qcc/venv/bin/python - <<'PY'
import json
from pathlib import Path

rows = []
for name in ("real64k", "composed-128k", "composed-256k", "composed-512k", "composed-1m"):
    path = Path(f"artifacts/g2-state-size-{name}-v1.json")
    if not path.exists():
        continue
    for row in json.loads(path.read_text())["rows"]:
        rows.append({"corpus": name, **row})
payload = {
    "schema": "ephemeral-kv-state-transfer-v1",
    "kind": "measured_primitive",
    "what": "the state a cold route transfers, in tokens and bytes, against raw history",
    "rows": rows,
}
Path("artifacts/g2-state-transfer-v1.json").write_text(json.dumps(payload, indent=2) + "\n")
for row in rows:
    print(f"{row['corpus']:<15} history {row['history_tokens']:>10,}  state {row['state_tokens']:>6,} tokens "
          f"{row['state_bytes']:>8,} bytes")
PY
