#!/bin/bash
GPUS=$1
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -m torch.distributed.run --nproc_per_node=$GPUS main.py ${@:2}
