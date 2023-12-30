#!/bin/bash


python slue_toolkit/eval/eval_w2v.py eval_ctc_model save/asr/w2v2-base-hvb --data manifest/slue-hvb --subset dev
python slue_toolkit/eval/eval_w2v.py eval_ctc_model save/asr/w2v2-base-hvb --data manifest/slue-hvb --subset test

