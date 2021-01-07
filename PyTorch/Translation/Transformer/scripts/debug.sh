# export CUDA_VISIBLE_DEVICES=1

RESULTS_DIR='/results'
CHECKPOINTS_DIR='/results/checkpoints'
STAT_FILE=${RESULTS_DIR}/run_log.json
mkdir -p $CHECKPOINTS_DIR

# nsys profile -t cuda \
#   -y 60 \
#   -d 20 \
#   -o transformer_baseline \
#   -f true \
#   -w true \


#nsys profile --stats=true -t cuda,nvtx \
#  -o transformer_baseline \
#  -f true \
  python /workspace/translation/train.py \
  /data/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens 5120 \
  --seed 1 \
  --max-epoch 1 \
  --no-epoch-checkpoints \
  --fuse-layer-norm \
  --amp \
  --amp-level O2 \
  --log-interval 10 \
  --save-dir ${RESULTS_DIR} \
  --stat-file ${STAT_FILE} \
