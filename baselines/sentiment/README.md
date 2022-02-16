## Sentiment analysis
#### Fine-tuning End-to-end model
Assuming that the preprocessed manifest files are in `manifest/slue-voxceleb` for SLUE-VoxPopuli. This command fine-tune a wav2vec 2.0 base model using one GPU.
```sh
bash baselines/sentiment/e2e_scripts/ft-w2v2-base-senti.sh manifest/slue-voxceleb save/sentiment/w2v2-base
```
#### Evaluation of End-to-end model
To evaluate the fine-tuned wav2vec 2.0 sentiment model, run following command or run `baselines/sentiment/e2e_scripts/eval.sh`
```sh
python slue_toolkit/eval/eval_w2v_sentiment.py --save-dir save/sentiment/w2v2-base --data manifest/slue-voxceleb --subset test
```

#### Fine-tuning NLP Topline
This command trains the deberta-large model on ground-truth text transcripts with raw labels
```sh
bash baselines/sentiment/nlp_scripts/ft-deberta-large-senti.sh
```

#### Evaluation of NLP Topline
To evaluate the fine-tuned nlp model, run following command or run `baselines/sentiment/nlp_scripts/eval.sh`
```sh
python slue_toolkit/eval/eval_nlp_sentiment.py --save-dir save/sentiment/nlp_topline_bert-base-cased --data manifest/slue-voxceleb --subset test
```

#### Training the Pipeline model
We don't fine-tune the model on ASR transcription.

#### Evaluating the Pipeline model

To evaluate the fine-tuned nlp model, run following command or run `baselines/sentiment/pipeline_scripts/eval.sh`

First, ASR transcription need to be prepared in manifest dir, and then evalution can be done using the same evaluation script with nlp topline.
```sh
python slue_toolkit/prepare/prepare_voxceleb_asr_pred.py --data manifest/slue-voxceleb --pred-data dataset/slue-voxceleb/preds/vc1/w2v2-large-lv60k-ft-slue-vc1-12h-lr1e-5-s1-mt800000-8gpu-update280000
python slue_toolkit/eval/eval_nlp_sentiment.py --save-dir save/sentiment/nlp_topline_bert-base-cased --data manifest/slue-voxceleb --subset test.asr-pred
```
