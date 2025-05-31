GPU='0,1'

# General Settings
MODEL_NAME='Meta-Llama-3-70B'
MODEL=meta-llama/$MODEL_NAME

SEED=0
TARGET_SPEEDUP=-1
NUM_SUBLAYERS_TO_REMOVE=52

EXP_NAME='run_sprint'
OUTPUT_DIR='../outputs/'${MODEL_NAME}/${EXP_NAME}
RES_FILE='summary.csv'
LOG='log_file'
PRUNED_MODEL_FILE='tuned_weights.pkl'

# Evaluation
# We do not support evaluation during pruning for 70B models
EVAL_STEPS='999' 
EVAL_SPEEDUPS=''
EVAL_ALL=False

# Importance scoring
BATCH=4
CPU_LAYERS=40
METRICS=l2_sentence

# Latency-aware importance scoring
LATENCY_AWARE=True

# Tunability-aware Sensitivity Evaluation
DAMP='0.'
DTYPE='float'
DIRECTION='out_channel'
CHANNEL_RATIO=0.75

# Avoiding Unnecessary Computations
NUM_CHECKPOINTS=8
NUM_CANDS=5
CHECKPOINTS_ON_CPU=True

cd src
CUDA_VISIBLE_DEVICES=$GPU python -u -m sprint \
    --model_name $MODEL \
    --seed $SEED \
    --num_remove_sublayers $NUM_SUBLAYERS_TO_REMOVE \
    --target_speedup $TARGET_SPEEDUP \
    --result_file $RES_FILE \
    --output_dir $OUTPUT_DIR \
    --logfile $LOG \
    --pruned_model_file $PRUNED_MODEL_FILE \
    --eval_every_step $EVAL_ALL \
    --eval_steps $EVAL_STEPS \
    --eval_speedups $EVAL_SPEEDUPS \
    --sensi_batch_size $BATCH \
    --metrics $METRICS \
    --num_cpu_layers $CPU_LAYERS \
    --latency_aware $LATENCY_AWARE \
    --tuning_sublayer 'MLP' \
    --in_comp_tuning True \
    --update_tuned_weights True \
    --num_candidate_sublayers $NUM_CANDS \
    --damping_coefficient $DAMP \
    --num_checkpoints $NUM_CHECKPOINTS \
    --checkpointing_on_cpu $CHECKPOINTS_ON_CPU \
    --in_comp_tuning_ratio $CHANNEL_RATIO
