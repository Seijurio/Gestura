#!/bin/zsh
# Télécharge les modèles MediaPipe utilisés par l'app (≈ 50 Mo) dans ./models
set -e
cd "$(dirname "$0")/.."
mkdir -p models
BASE="https://storage.googleapis.com/mediapipe-models"
typeset -A MODELS
MODELS=(
  hand_landmarker.task        "$BASE/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
  face_landmarker.task        "$BASE/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
  selfie_segmenter.tflite     "$BASE/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
  pose_landmarker_full.task   "$BASE/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
  pose_landmarker_heavy.task  "$BASE/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
for name url in ${(kv)MODELS}; do
  if [ ! -s "models/$name" ]; then
    echo "↓ $name"
    curl -sSfL -o "models/$name" "$url"
  fi
done
echo "Modèles prêts."
