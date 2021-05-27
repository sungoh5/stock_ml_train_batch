#!/bin/bash
set -e

function read_version() {
  VERSION=`cat .version`
}
read_version
IMAGE_NAME='stock-ml-training-batch'
echo  "Build [${IMAGE_NAME}] image - version : ${VERSION}"

docker build --tag ${IMAGE_NAME}:${VERSION} .
