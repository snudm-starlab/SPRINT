GPU='0,1'

MODEL_NAME='Meta-Llama-3-70B'
MODEL=meta-llama/$MODEL_NAME

EXP_NAME='run_sprint'
OUTPUT_DIR='../outputs/'${MODEL_NAME}/${EXP_NAME}
RES_DIR=$OUTPUT_DIR
LOG_DIR=$OUTPUT_DIR
PRUNED_MODEL_DIR=$OUTPUT_DIR

PRUNED_MODEL_FILE='tuned_weights.pkl'
NUM_PRUNED_SUBLAYERS=2

LOG_FILE='eval_log_file'_${NUM_PRUNED_SUBLAYERS}
cd src
CUDA_VISIBLE_DEVICES=$GPU python -u -m load_and_evaluate \
    --model_name $MODEL \
    --log_dir $LOG_DIR \
    --log_file $LOG_FILE \
    --pruned_model_dir $PRUNED_MODEL_DIR \
    --pruned_model_file $PRUNED_MODEL_FILE \
    --num_pruned_sublayers $NUM_PRUNED_SUBLAYERS \
    --load_weight True

