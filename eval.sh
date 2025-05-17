#!/bin/bash

# CaptioningVLM 모델 평가를 위한 쉘 스크립트
# 사용법: ./run_eval.sh [데이터셋] [모델_경로] [비디오_경로]

# 기본값 설정
DATASET=${1:-"NextQA"}
MODEL_PATH=${2:-"outputs/captionvlm/checkpoint-1000/model.pt"}
VIDEO_ROOT=${3:-"DATAS/eval/NextQA/NExTVideo"}
RESULTS_DIR="results/captionvlm_eval/${DATASET}"
DATA_PATH=""

# 데이터셋에 따른 경로 설정
case $DATASET in
  "NextQA")
    DATA_PATH="DATAS/eval/NextQA/formatted_dataset_val.json"
    ;;
  *)
    echo "지원되지 않는 데이터셋: $DATASET"
    echo "지원 데이터셋: VideoMME, MovieChat, VSI"
    exit 1
    ;;
esac

# 필요한 디렉토리 생성
mkdir -p $RESULTS_DIR

# GPU 개수 확인
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "사용 가능한 GPU 수: $NUM_GPUS"

# 멀티프로세스 사용 여부 결정
if [ $NUM_GPUS -gt 1 ]; then
  MULTIPROCESS_FLAG="--multiprocess"
else
  MULTIPROCESS_FLAG=""
fi

# 평가 실행
echo "평가 시작: $DATASET"
echo "모델 디렉토리: $MODEL_DIR"
echo "비디오 경로: $VIDEO_ROOT"
echo "결과 디렉토리: $RESULTS_DIR"

python evaluation.py \
  --model_path $MODEL_DIR \
  --dataset_name $DATASET \
  --data_path $DATA_PATH \
  --video_root $VIDEO_ROOT \
  --results_dir $RESULTS_DIR \
  --max_frames_num 64 \
  --max_new_tokens 100 \
  --use_time_ins \
  --calc_acc \
  $MULTIPROCESS_FLAG

echo "평가 완료"
echo "결과: $RESULTS_DIR/results.json"