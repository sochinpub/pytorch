#!/bin/bash

docker run -d --name pytorch-dev -it --gpus all -v /root/sochin:/root/sochin --entrypoint bash torch-build:0.0.2 -c "while true; do sleep 30; done;"

