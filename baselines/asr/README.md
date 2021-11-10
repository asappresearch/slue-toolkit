### ASR
#### Fine-tuning
Assuming that the preprocessed manifest files are in `manifest/slue-voxceleb` and `manifest/slue-voxpopuli` for SLUE-VoxCeleb and SLUE-VoxPopuli. This command fine-tune a wav2vec 2.0 base model on these two datasets using one GPU.
```sh
bash baselines/asr/ft-w2v2-base.sh manifest/slue-voxceleb save/asr/w2v2-base-vc
bash baselines/asr/ft-w2v2-base.sh manifest/slue-voxpopuli save/asr/w2v2-base-vp
```

To fine-tune wav2vec 2.0 large ll60k model using 8 GPUs, please run:
```sh
bash baselines/asr/ft-w2v2-large.sh manifest/slue-voxceleb save/asr/w2v2-large-vc
bash baselines/asr/ft-w2v2-large.sh manifest/slue-voxpopuli save/asr/w2v2-large-vp
```

#### Evaluation
To evaluate the fine-tuned wav2vec 2.0 ASR models on the dev set, please run the following commands.
```sh
python slue_toolkit/eval/eval_w2v.py eval_asr save/asr/w2v2-base-vc --data manifest/slue-voxceleb --subset dev
python slue_toolkit/eval/eval_w2v.py eval_asr save/asr/w2v2-base-vp --data manifest/slue-voxpopuli --subset dev
```
The WER will be printed directly.
The predictions are saved in `save/asr/w2v2-base-vc/pred-dev.wrd` and `save/asr/w2v2-base-vp/pred-dev.wrd` and can be used for pipeline models.

Instead, we can also dump the predictions of all subsets into `preds/vc/w2v2-base-vc` folder using this command.
```sh
python slue_toolkit/eval/eval_w2v.py dump_asr_preds save/asr/w2v2-base-vc --task vc
python slue_toolkit/eval/eval_w2v.py dump_asr_preds save/asr/w2v2-base-vp --task vp
```

To decode with LM, please first build the tri-gram LM using this command (this downloads TED-LIUM 3 data which takes ~56GB)
```sh
bash scripts/build_t3_lm.sh $kenlm_build_bin
```
where `$kenlm_build_bin` is the path of your kenlm build folder (e.g., `/home/user/kenlm/build/bin`).
After the tri-gram LM is trained, we can decode with it using these commands
```sh
python slue_toolkit/eval/eval_w2v.py eval_asr save/asr/w2v2-base-vc --data manifest/slue-voxceleb --subset dev --lm t3/3
python slue_toolkit/eval/eval_w2v.py eval_asr save/asr/w2v2-base-vp --data manifest/slue-voxpopuli --subset dev --lm t3/3
```
