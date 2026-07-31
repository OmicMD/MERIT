#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$MODEL_DIR")"

echo "=== Model Test ==="
echo ""

# Step 1: Serialize model bundle (if not already done)
BUNDLE="$MODEL_DIR/model_bundle.pkl"
if [ ! -f "$BUNDLE" ]; then
    echo "[1/3] Serializing model bundle..."
    python "$MODEL_DIR/serialize_model_bundle.py"
else
    echo "[1/3] Model bundle exists, skipping serialization"
fi

# Step 2: Run prediction locally
echo ""
echo "[2/3] Running prediction on test compounds..."
python "$MODEL_DIR/predict.py" \
    --input-dir "$SCRIPT_DIR/test_input" \
    --output "$SCRIPT_DIR/test_output.csv" \
    --bundle "$BUNDLE"

# Step 3: Validate output
echo ""
echo "[3/3] Validating output..."
OUTPUT_CSV="$SCRIPT_DIR/test_output.csv"
python3 -c "
import pandas as pd, sys
df = pd.read_csv('${OUTPUT_CSV}')
errors = []
if len(df) != 3:
    errors.append(f'Expected 3 rows, got {len(df)}')
for col in ['zinc_id', 'disease', 'P_FAIL_SAFETY', 'P_FAIL_EFFICACY', 'P_PASS']:
    if col not in df.columns:
        errors.append(f'Missing column: {col}')
for col in ['P_FAIL_SAFETY', 'P_FAIL_EFFICACY', 'P_PASS']:
    if col in df.columns:
        if df[col].min() < 0 or df[col].max() > 1:
            errors.append(f'{col} out of [0,1] range')
if errors:
    print('VALIDATION FAILED:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print('VALIDATION PASSED')
    print(f'  3 compounds, probabilities in [0,1]')
    for _, r in df.iterrows():
        print(f\"  {r['zinc_id']}: P(safe)={1-r['P_FAIL_SAFETY']:.1%}, P(effic)={1-r['P_FAIL_EFFICACY']:.1%}, P(pass)={r['P_PASS']:.1%}\")
"

echo ""
echo "=== Test Complete ==="
