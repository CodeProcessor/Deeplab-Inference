#!/usr/bin/env bash

docker run -it --rm -p 80:80 -v $(pwd)/docker_output:/home/output deeplab:v1.1
