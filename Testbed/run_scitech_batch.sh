#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-/tmp/orsini_scitech_lp_batch}"
TRACE_IDS=(
  lp_15sccm_10a1
  lp_15sccm_15a1
  lp_15sccm_20a1
  lp_15sccm_25a1
)

cd "$ROOT_DIR"

mkdir -p "$OUTPUT_DIR"

for TRACE_ID in "${TRACE_IDS[@]}"; do
  python3 -m orsini_lp.cli compare sample_data/local_manifest.csv \
    --trace-id "$TRACE_ID" \
    --output-dir "$OUTPUT_DIR/$TRACE_ID" \
    --nlive 40 \
    --dlogz 2.0 \
    --posterior-draws 40 \
    --max-points 80 \
    --random-seed 9
done

python3 - <<'PY' "$OUTPUT_DIR" "${TRACE_IDS[@]}"
from pathlib import Path
import sys
import pandas as pd

output_dir = Path(sys.argv[1])
trace_ids = sys.argv[2:]
rows = []

for trace_id in trace_ids:
    path = output_dir / trace_id / "comparison_summary.csv"
    frame = pd.read_csv(path)
    row = {"trace_id": trace_id}
    for _, item in frame.iterrows():
        metric = item["metric"]
        row[f"{metric}_bayes_median"] = item["bayes_median"]
        row[f"{metric}_legacy_value"] = item["legacy_value"]
        row[f"{metric}_delta"] = item["delta"]
    rows.append(row)

pd.DataFrame(rows).to_csv(output_dir / "batch_summary.csv", index=False)
PY

find "$OUTPUT_DIR" -maxdepth 2 -type f | sort

echo
echo "Batch outputs written to: $OUTPUT_DIR"
