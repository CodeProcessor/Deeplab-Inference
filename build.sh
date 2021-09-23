#!/usr/bin/env bash

IMAGE_NAME=deeplab:v1.1
#docker rmi $IMAGE_NAME
docker build -t $IMAGE_NAME .
