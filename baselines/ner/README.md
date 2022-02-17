### NER
#### Fine-tuning End-to-end model
Assuming that the preprocessed manifest files are in `manifest/slue-voxpopuli` for SLUE-VoxPopuli. This command fine-tune a wav2vec 2.0 base model using one GPU.
```sh
bash baselines/ner/e2e_scripts/ft-w2v2-base.sh manifest/slue-voxpopuli/e2e_ner save/e2e_ner/w2v2-base
```

To fine-tune wav2vec 2.0 large ll60k model using 8 GPUs, please run:
```sh
bash baselines/ner/e2e_scripts/ft-w2v2-base.sh manifest/slue-voxpopuli/e2e_ner save/e2e_ner/w2v2-large
```

#### Evaluating End-to-End model
To decode with LM, please first build the 4-gram LM using this command
```sh
bash scripts/build_vp_ner_lm.sh $kenlm_build_bin
```
where `$kenlm_build_bin` is the path of your kenlm build folder (e.g., `/home/user/kenlm/build/bin`).

To evaluate the fine-tuned wav2vec 2.0 E2E NER model on the dev set, please run the following command.
1. Decoded without language model
```sh
bash baselines/ner/e2e_scripts/eval-ner.sh w2v2-base dev combined nolm
```

2. Decoded with language model
```sh
bash baselines/ner/e2e_scripts/eval-ner.sh w2v2-base dev combined vp_ner/4
```

#### Fine-tuning NLP Topline
This command trains the deberta-base model on ground-truth text transcripts with raw labels
```sh
bash baselines/ner/nlp_scripts/ft-deberta.sh deberta-base raw
```
The above command can also be used to train `deberta-large` model and also accepts `combined` tag set as argument

#### Evaluation of NLP Topline
The following command evaluates the trained deberta-base model on dev set with combined labels
```sh
bash baselines/ner/nlp_scripts/eval-deberta.sh deberta-base dev combined
```

#### Training the Pipeline model
The ASR module is trained using the scripts mentioned [here](https://github.asapp.dev/ASAPPinc/slue-toolkit/tree/master#fine-tuning).
The text NER module is trained using the scripts mentioned [here](https://github.asapp.dev/ASAPPinc/slue-toolkit/tree/master#fine-tuning-1).

#### Evaluating the Pipeline model
The following command evalutes the pipeline model that uses w2v2-base as ASR backbone and deberta-base as text NER backbone, with the former decoded using the T3 language model as mentioned [here](https://github.asapp.dev/ASAPPinc/slue-toolkit/blob/master/README.md#evaluation-1).
```sh
bash baselines/ner/pipeline_scripts/eval.sh w2v2-base deberta-base dev combined t3/3
```