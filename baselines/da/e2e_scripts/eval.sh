#!/bin/bash

python slue_toolkit/eval/eval_w2v_da.py \
	--manifest-dir manifest/slue-hvb \
	--split test \
	--save-dir save/da/w2v2-base-da_baseline_0 \
	--use-gpu \
	--eval &
