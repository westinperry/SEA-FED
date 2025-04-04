#!/bin/bash

# Configuration
ROUNDS=1
EPOCH=1
BASE_PATH="/media/westin/4EDCAC80DCAC6445/FAD/models"
DATASET=UCSD_P2_256
MODEL_NAME="AE"   # AE or Gated_AE

# Iterate over rounds
for ROUND in $(seq 1 $ROUNDS); do
    PREV=$((ROUND - 1))
    echo "==== Starting Round $ROUND ===="

    # Training Phase
    if [ $ROUND -eq 1 ]; then
        for CLIENT in 1 2 3 4; do
            echo "Training client $CLIENT from scratch"
            python script_training.py \
                --ModelRoot "${BASE_PATH}/client_${CLIENT}/" \
                --OutputFile "${BASE_PATH}/client_${CLIENT}/client${CLIENT}_local1.pt" \
                --EpochNum $EPOCH \
                --BatchSize 6 \
                --DataRoot "../datasets/processed_${CLIENT}/" \
                --Dataset $DATASET \
                --TextLogInterval 10 \
                --IsTbLog False \
                --ModelName $MODEL_NAME
        done
    else
        for CLIENT in 1 2 3 4; do
            RESUME_PATH="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${PREV}.pt"
            if [ ! -f "$RESUME_PATH" ]; then
                echo "Error: Resume path $RESUME_PATH does not exist!"
                exit 1
            fi
            echo "Training client $CLIENT resuming from $RESUME_PATH"
            python script_training.py \
                --ModelRoot "${BASE_PATH}/client_${CLIENT}/" \
                --OutputFile "${BASE_PATH}/client_${CLIENT}/client${CLIENT}_local${ROUND}.pt" \
                --ResumePath "$RESUME_PATH" \
                --EpochNum $EPOCH \
                --BatchSize 6 \
                --DataRoot "../datasets/processed_${CLIENT}/" \
                --Dataset $DATASET \
                --TextLogInterval 10 \
                --IsTbLog False \
                --IsResume \
                --ModelName $MODEL_NAME
        done
    fi

    # Federated Averaging Phase
    INPUT_PATHS=""
    OUTPUT_PATHS=""
    for CLIENT in 1 2 3 4; do
        INPUT_PATH="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_local${ROUND}.pt"
        if [ ! -f "$INPUT_PATH" ]; then
            echo "Error: Input path $INPUT_PATH does not exist!"
            exit 1
        fi
        OUTPUT_PATH="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${ROUND}.pt"
        INPUT_PATHS="$INPUT_PATHS $INPUT_PATH"
        OUTPUT_PATHS="$OUTPUT_PATHS $OUTPUT_PATH"
    done
    echo "Performing federated averaging with inputs: $INPUT_PATHS"
    python script_fedavg.py \
        --input-paths $INPUT_PATHS \
        --output-paths $OUTPUT_PATHS \
        --ModelName $MODEL_NAME

    # Testing Phase
    echo "Testing clients for Round $ROUND"
    # Write results into the ../results folder (assumes 'results' is a sibling of 'scripts')
    echo "Round $ROUND" >> ../results/results.txt
    for CLIENT in 1 2 3 4; do
        MODEL_PATH="${BASE_PATH}/client_${CLIENT}/client${CLIENT}_combined${ROUND}.pt"
        if [ ! -f "$MODEL_PATH" ]; then
            echo "Error: Model path $MODEL_PATH does not exist!"
            exit 1
        fi
        echo "Testing client $CLIENT with model $MODEL_PATH"
        python script_testing.py \
            --ModelFilePath "$MODEL_PATH" \
            --DataRoot "../datasets/processed_${CLIENT}/" \
            --Dataset $DATASET \
            --ModelName $MODEL_NAME
    done
    echo "==== Finished Round $ROUND ===="
done

