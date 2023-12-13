#!/bin/bash
cd BLIP
python -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=4 train_retrieval_vg.py --train-data PATH_TO_LAION --vg-data PATH_TO_VG 