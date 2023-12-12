#!/bin/bash
cd BLIP
python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=1 train_retrieval_vg.py --evaluate PATH_TO-PRETRAINED_CHECKPOINT --winoground --vlchecklist --vsr