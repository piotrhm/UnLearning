BASE_OUTPUT="/home/gmhelm/repo/UnLearning/output"

# Glob and sort to have a stable order
mapfile -t EXP_DIRS < <(ls -d "${BASE_OUTPUT}"/* | sort)

for EXP_DIR in $EXP_DIRS; do
  echo "[$(date)] Using EXP_DIR=$EXP_DIR"
  SAMPLES_DIR=$EXP_DIR/images
  METRICS_DIR=$EXP_DIR/metrics

  mkdir -p "${METRICS_DIR}"

  for class_name in $SAMPLES_DIR/*; do
    class_name=$(basename $class_name)
    python compute_metrics.py -m acc --prompts_json data/cat.json --samples_dir "$SAMPLES_DIR/$class_name" --output_dir "$METRICS_DIR/$class_name"
  done
done