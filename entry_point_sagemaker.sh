#!/bin/bash
set -e

if [ -f /opt/ml/input/data/train/swde_ae_preprocessed.zip ]; then
  echo "🗜️  Unzipping dataset…"
  unzip -q /opt/ml/input/data/train/swde_ae_preprocessed.zip -d /opt/ml/input/data/train/
fi

pip install -r requirements_sagemaker.txt

python train_sagemaker.py "$@"
