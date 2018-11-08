#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./pretrained_models/resnet_v101"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Download pre-trained model
BASE_URL="http://download.tensorflow.org/models"
FILENAME="resnet_v2_101_2017_04_14.tar.gz"
echo "Downloading ${FILENAME} to ${WORK_DIR}"
wget -nd -c "http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz"

# Unzip model file
echo "Uncompressing ${FILENAME}"
tar -xf "${FILENAME}"