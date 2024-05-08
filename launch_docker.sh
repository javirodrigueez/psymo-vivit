#!/bin/bash
docker run --gpus '"device=0,1"' --rm -it \
	--name jrodriguez_python \
	--volume="/home/jrodriguez/misc/psycho-gait:/workspace:rw" \
	--volume="/home/jrodriguez/misc/gait-data:/data:rw" \
	--shm-size=16gb \
	--memory=16gb \
	jrodriguez/python bash
