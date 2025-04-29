#!/bin/bash

# ┌─────────────────────────────────────────────────────┐
# │           GLOBAL CONFIGURATION                      │
# └─────────────────────────────────────────────────────┘
ROUNDS=10
EPOCH=1
BASE_PATH="../models"
DATASET="UCSD_P2_256"
MODEL_NAME="AE"      # AE or Gated_AE
BATCH_SIZE=6
TEXT_LOG_INT=10
TB_LOG=False
START_ROUND=1

PYTHON_BIN=$(which python)

# ┌─────────────────────────────────────────────────────┐
# │               MAIN LOOP                              │
# └─────────────────────────────────────────────────────┘
for ROUND in $(seq $START_ROUND $ROUNDS); do
    echo "==== Starting Round $ROUND ===="

    # -----------------------
    # 1) Local Training
    # -----------------------
    for CLIENT in 1 2 3 4; do
        if [ "$ROUND" -eq 1 ]; then
            echo "→ [Train] Client $CLIENT round $ROUND from scratch"
            "$PYTHON_BIN" script_training.py \
                --Mode train \
                --ModelRoot "${BASE_PATH}/client_${CLIENT}" \
                --OutputFile "client${CLIENT}_local${ROUND}.pt" \
                --EpochNum $EPOCH \
                --BatchSize $BATCH_SIZE \
                --DataRoot "../datasets/processed_${CLIENT}" \
                --Dataset $DATASET \
                --TextLogInterval $TEXT_LOG_INT \
                --IsTbLog $TB_LOG \
                --ModelName $MODEL_NAME \
                --PlotGraph True \
                --Round $ROUND \
                --ClientID $CLIENT \
                --IsSaveSEAdapter True
        else
            PREV_ROUND=$((ROUND - 1))
            RESUME="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${PREV_ROUND}.pt"
            if [ ! -f "$RESUME" ]; then
                echo "❌ Missing $RESUME" >&2
                exit 1
            fi
            echo "→ [Train] Client $CLIENT round $ROUND resuming from $RESUME"
            "$PYTHON_BIN" script_training.py \
                --Mode train \
                --ModelRoot "${BASE_PATH}/client_${CLIENT}" \
                --OutputFile "client${CLIENT}_local${ROUND}.pt" \
                --ResumePath "$RESUME" \
                --IsResume \
                --EpochNum $EPOCH \
                --BatchSize $BATCH_SIZE \
                --DataRoot "../datasets/processed_${CLIENT}" \
                --Dataset $DATASET \
                --TextLogInterval $TEXT_LOG_INT \
                --IsTbLog $TB_LOG \
                --ModelName $MODEL_NAME \
                --PlotGraph True \
                --Round $ROUND \
                --ClientID $CLIENT \
                --IsSaveSEAdapter True
        fi
    done

    # -----------------------
    # 2) Federated Averaging
    # -----------------------
    echo "→ [FedAvg] aggregating clients’ models"
    INPUT_PATHS=()
    OUTPUT_PATHS=()
    for CLIENT in 1 2 3 4; do
        inp="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_local${ROUND}.pt"
        out="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${ROUND}.pt"
        if [ ! -f "$inp" ]; then
            echo "❌ Missing $inp" >&2
            exit 1
        fi
        INPUT_PATHS+=("$inp")
        OUTPUT_PATHS+=("$out")
    done

    "$PYTHON_BIN" script_fedavg.py \
        --input-paths ${INPUT_PATHS[*]} \
        --output-paths ${OUTPUT_PATHS[*]} \
        --ModelName $MODEL_NAME \
        --Channels 1

    # -----------------------
    # 2b) Save features from Combined Models
    # -----------------------
    echo "→ [Save] SE/Adapter/Latents from Combined Models (round $ROUND)"

    for CLIENT in 1 2 3 4; do
        MODEL="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${ROUND}.pt"
        if [ ! -f "$MODEL" ]; then
            echo "❌ Missing $MODEL" >&2
            exit 1
        fi

        echo "   • Client $CLIENT → Saving Combined Model Features"
        "$PYTHON_BIN" script_save_features.py \
            --Mode eval \
            --ModelFilePath "$MODEL" \
            --DataRoot "../datasets/processed_${CLIENT}" \
            --Dataset $DATASET \
            --ModelName $MODEL_NAME \
            --Round $ROUND \
            --ClientID $CLIENT
    done

    # -----------------------
    # 3) Testing
    # -----------------------
    # echo "→ [Test] evaluating aggregated models (round $ROUND)"
    # echo "Round $ROUND" >> ../results/results.txt

    # if [ "$MODEL_NAME" == "AE" ]; then
    #     CLIENTS=(1)
    # else
    #     CLIENTS=(1 2 3 4)
    # fi

    # for CLIENT in "${CLIENTS[@]}"; do
    #     MODEL="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${ROUND}.pt"
    #     if [ ! -f "$MODEL" ]; then
    #         echo "❌ Missing $MODEL" >&2
    #         exit 1
    #     fi
    #     echo "   • Client $CLIENT → $MODEL"
    #     "$PYTHON_BIN" script_testing.py \
    #         --Mode eval \
    #         --ModelFilePath "$MODEL" \
    #         --DataRoot "../datasets/processed_${CLIENT}" \
    #         --Dataset $DATASET \
    #         --ModelName $MODEL_NAME \
    #         --Round $ROUND
    # done

    echo "==== Finished Round $ROUND ===="
done
