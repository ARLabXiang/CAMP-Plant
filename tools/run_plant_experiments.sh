#!/bin/bash
# =============================================================================
# run_plant_experiments.sh
# =============================================================================
# Runs all plant-growth prediction experiments from the CAMP paper, including
# the original 4 baselines (ConvLSTM, PredRNN, PhyDNet, SimVP), the proposed
# CAMP model, and 2 additional baselines (MIM, TAU).
#
# Compatible with macOS default bash 3.2 (no associative arrays used).
#
# Usage:
#   # Run on KOMATSUNA dataset
#   bash tools/run_plant_experiments.sh komatsuna /path/to/testdata/data
#
#   # Run on Arabidopsis dataset
#   bash tools/run_plant_experiments.sh arabidopsis /path/to/testdata/water
#
#   # Run a single method for debugging
#   bash tools/run_plant_experiments.sh komatsuna /path/to/data camp
#
# Arguments:
#   $1  DATASET   : 'komatsuna' or 'arabidopsis'
#   $2  DATA_ROOT : path to the dataset root directory
#   $3  METHOD    : (optional) run only this method; default = all
#   $4  EPOCHS    : (optional) number of training epochs; default = 200
#
# Examples:
#   # Quick test run with 3 epochs
#   bash tools/run_plant_experiments.sh komatsuna /path/to/data all 3
#
#   # Single method, custom epochs
#   bash tools/run_plant_experiments.sh komatsuna /path/to/data camp 50
#
# Outputs are written to:
#   work_dirs/{DATASET}_{METHOD}/
#
# After all models finish, POI evaluation is run automatically on each.
# =============================================================================

set -e  # Exit immediately on any error

# Allow PyTorch MPS backend to fall back to CPU for any unsupported op
# (safe no-op on CUDA / CPU machines)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Resolve the project root (parent of the tools/ directory this script lives in)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Pick the Python interpreter: prefer the project's .venv if it exists, then
# fall back to whatever python3 is on PATH.  This means the script works both
# when the .venv is active AND when called from a bare shell.
if [ -x "${SCRIPT_DIR}/.venv/bin/python3" ]; then
    PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
else
    PYTHON="python3"
fi

# Ensure the project root is always on PYTHONPATH so openstl is importable.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# --------------------------------------------------------------------------- #
# Parse arguments                                                             #
# --------------------------------------------------------------------------- #
DATASET=${1:-komatsuna}
DATA_ROOT=${2:-./data}
SINGLE_METHOD=${3:-all}   # if provided, run only this method
EPOCHS=${4:-200}          # number of training epochs (default: 200)

if [ "$DATASET" != "komatsuna" ] && [ "$DATASET" != "arabidopsis" ]; then
    echo "ERROR: DATASET must be 'komatsuna' or 'arabidopsis'"
    exit 1
fi

echo "============================================================"
echo "  CAMP Plant Growth Prediction Experiments"
echo "  Dataset  : $DATASET"
echo "  Data root: $DATA_ROOT"
echo "  Epochs   : $EPOCHS"
echo "============================================================"

# --------------------------------------------------------------------------- #
# Lookup helpers (replaces bash 4 associative arrays)                         #
# --------------------------------------------------------------------------- #

# Return the config file path for a given short method name
get_config() {
    case "$1" in
        convlstm)     echo "configs/plant/ConvLSTM.py"    ;;
        predrnn)      echo "configs/plant/PredRNN.py"     ;;
        phydnet)      echo "configs/plant/PhyDNet.py"     ;;
        simvp)        echo "configs/plant/SimVP.py"       ;;
        mim)          echo "configs/plant/MIM.py"         ;;
        tau)          echo "configs/plant/TAU.py"         ;;
        camp)         echo "configs/plant/CAMP.py"        ;;
        # Ablation study variants (Reviewer 1, Comment 3)
        camp_base)    echo "configs/plant/CAMP_base.py"   ;;
        camp_no_cls)  echo "configs/plant/CAMP_no_cls.py" ;;
        camp_full)    echo "configs/plant/CAMP_full.py"   ;;
        *)            echo "" ;;
    esac
}

# Return the method_map key (plant_ prefix for baselines, plain for CAMP)
get_method_key() {
    case "$1" in
        convlstm)     echo "plant_convlstm" ;;
        predrnn)      echo "plant_predrnn"  ;;
        phydnet)      echo "plant_phydnet"  ;;
        simvp)        echo "plant_simvp"    ;;
        mim)          echo "plant_mim"      ;;
        tau)          echo "plant_tau"      ;;
        camp)         echo "camp"           ;;
        # Ablation variants — method key matches method_maps entry
        camp_base)    echo "camp_base"      ;;
        camp_no_cls)  echo "camp_no_cls"    ;;
        camp_full)    echo "camp_full"      ;;
        *)            echo "" ;;
    esac
}

# --------------------------------------------------------------------------- #
# Helper: train + test a single method                                        #
# --------------------------------------------------------------------------- #
run_method() {
    local METHOD_SHORT=$1
    local CONFIG
    local METHOD_KEY
    CONFIG=$(get_config "$METHOD_SHORT")
    METHOD_KEY=$(get_method_key "$METHOD_SHORT")
    local EX_NAME="${DATASET}_${METHOD_SHORT}_ep${EPOCHS}"
    local SAVE_DIR="work_dirs/${EX_NAME}"
    local LOG_DIR="work_dirs/${EX_NAME}/terminal_logs"
    local LOG_FILE="${LOG_DIR}/train_terminal_$(date '+%Y%m%d_%H%M%S').log"

    mkdir -p "$LOG_DIR"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Training: $METHOD_SHORT on $DATASET"
    echo "  Config  : $CONFIG"
    echo "  Output  : $SAVE_DIR"
    echo "  Log     : $LOG_FILE"
    echo "------------------------------------------------------------"

    # tee writes to both terminal and log file
    {
        echo "===== Training: $METHOD_SHORT on $DATASET ====="
        echo "  Config   : $CONFIG"
        echo "  Save dir : $SAVE_DIR"
        echo "  Epochs   : $EPOCHS"
        echo "  Started  : $(date)"
        echo ""

        START_TIME=$(date +%s)

        "$PYTHON" tools/train.py \
            --method       "$METHOD_KEY" \
            --dataname     "$DATASET" \
            --data_root    "$DATA_ROOT" \
            --config_file  "$CONFIG" \
            --ex_name      "$EX_NAME" \
            --res_dir      "work_dirs" \
            --epoch        "$EPOCHS" \
            --overwrite

        END_TIME=$(date +%s)
        ELAPSED=$(( END_TIME - START_TIME ))
        echo ""
        echo "  Training finished at : $(date)"
        echo "  Total training time  : ${ELAPSED}s  ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"
        echo ""
        echo "  Running POI evaluation..."

        if [ -f "work_dirs/${EX_NAME}/saved/preds.npy" ]; then
            "$PYTHON" tools/eval_poi.py \
                --pred_dir  "work_dirs/${EX_NAME}/saved" \
                --threshold 0.1 \
                --plot
        else
            echo "  WARNING: preds.npy not found; skipping POI evaluation."
        fi

        echo ""
        echo "  Done: $METHOD_SHORT"
        echo "===== End: $METHOD_SHORT ====="
    } 2>&1 | tee "$LOG_FILE"
}

# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #
if [ "$SINGLE_METHOD" = "all" ]; then
    # Baselines first, then CAMP and its ablation variants
    for METHOD_SHORT in convlstm predrnn phydnet simvp mim tau \
                        camp camp_base camp_no_cls camp_full; do
        run_method "$METHOD_SHORT"
    done
else
    CONFIG=$(get_config "$SINGLE_METHOD")
    if [ -z "$CONFIG" ]; then
        echo "ERROR: Unknown method '$SINGLE_METHOD'."
        echo "Valid options: convlstm predrnn phydnet simvp mim tau camp"
        exit 1
    fi
    run_method "$SINGLE_METHOD"
fi

# --------------------------------------------------------------------------- #
# Summary table  (printed to terminal AND saved as CSV)                        #
# --------------------------------------------------------------------------- #
CSV_FILE="work_dirs/${DATASET}_ep${EPOCHS}_results.csv"

echo ""
echo "============================================================"
echo "  Results Summary — $DATASET  (epoch=$EPOCHS)"
echo "============================================================"
echo ""

# Write CSV header
echo "method,mae,mse,ssim,psnr,poi_mae,new_poi_mae,miss_poi_mae" > "$CSV_FILE"

for METHOD_SHORT in convlstm predrnn phydnet simvp mim tau \
                    camp camp_base camp_no_cls camp_full; do
    METRICS_FILE="work_dirs/${DATASET}_${METHOD_SHORT}_ep${EPOCHS}/saved/metrics.npy"
    POI_FILE="work_dirs/${DATASET}_${METHOD_SHORT}_ep${EPOCHS}/saved/poi_results.npy"

    if [ -f "$METRICS_FILE" ]; then
        # Print to terminal and append one row to the CSV in a single Python call
        "$PYTHON" -c "
import numpy as np

method  = '$METHOD_SHORT'
metrics = np.load('$METRICS_FILE')   # [mae, mse, ssim, psnr] or [mae, mse]
mae  = float(metrics[0])
mse  = float(metrics[1])
ssim = float(metrics[2]) if len(metrics) > 2 else float('nan')
psnr = float(metrics[3]) if len(metrics) > 3 else float('nan')

poi_mae  = float('nan')
new_mae  = float('nan')
miss_mae = float('nan')
try:
    poi = np.load('$POI_FILE', allow_pickle=True).item()
    poi_mae  = float(poi.get('poi_mae',  float('nan')))
    new_mae  = float(poi.get('new_mae',  float('nan')))
    miss_mae = float(poi.get('miss_mae', float('nan')))
except Exception:
    pass

# Terminal output
print(f'  {method:<14s} | MAE={mae:8.2f} | MSE={mse:8.2f}', end='')
if not (poi_mae != poi_mae):   # nan check
    print(f' | POI-MAE={poi_mae:8.2f}', end='')
print()

# CSV row  (nan → empty string for clean spreadsheet import)
def fmt(v):
    return '' if v != v else f'{v:.4f}'   # nan != nan is True

with open('$CSV_FILE', 'a') as f:
    f.write(f'{method},{fmt(mae)},{fmt(mse)},{fmt(ssim)},{fmt(psnr)},'
            f'{fmt(poi_mae)},{fmt(new_mae)},{fmt(miss_mae)}\n')
"
    fi
done

echo ""
echo "============================================================"
echo "  All experiments complete."
echo "  Full results : work_dirs/"
echo "  CSV summary  : $CSV_FILE"
echo "============================================================"
