#!/usr/bin/env bash

IMAGE_NAME=deeplab:v1.0
docker rmi $IMAGE_NAME
docker build -t $IMAGE_NAME .
