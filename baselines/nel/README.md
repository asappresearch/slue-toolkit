## NEL

### Notes on NEL training
The baseline NEL models are built on top of the baseline NER models. No separate training is done for NEL. 

### Notes on NEL evaluation
For evaluation, two hyperparameters are tuned on dev set: *offset* and *incl_blank*. *offset* is a fixed duration by which we shift the time stamp predictions. *incl_blank* is a Boolean to decide whether the trailing blank tokens in the CTC emissions are considered as a part of the predicted segment. When *incl_blank* is `True`, the segment between the start and end word separator tokens is considered as hypothesis. 

*word-F1* metric is evaluated with a *tolerance* hyperparameter. *tolerance*, a value between 0 and 1, is the fraction of overlap between a ground-truth word segment and the predicted region needed to count the word as detected; ρ = 1 means a perfect match is required to count the word as detected.

<span style="color:red">Add figure 2 from the paper</span>.


### End-to-end model

Time-stamps are extracted using CTC emissions from the [E2E NER model](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/ner#fine-tuning-end-to-end-model). The frames between start and end special characters constitute the detected entity segment.

<u>Step 1: Extract CTC emissions from E2E NER model and save character-level timestamps.</u>
```
bash baselines/nel/decode.sh e2e_ner dev
```

<u>Step 2: Hyperparameter search on dev.</u>
```
bash baselines/nel/eval_nel.sh e2e
```

### Pipeline model

Time-stamps are extracted using CTC emissions from the ASR model. The frames correponding to the entity phrase as detected by the [text NER model](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/ner#fine-tuning-nlp-topline) constitute the detected entity segment. 

<span style="color:red">Add evaluation scripts in the table format</span>.

#### pipeline-w2v2

ASR model: [wav2vec2.0 finetuned for ASR](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/asr)
text NER model: [DeBERTa-Base finetuned for NER](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/ner#fine-tuning-nlp-topline)

<u>Step 1: Extract CTC emissions from ASR model and save character-level timestamps.</u>
```
bash baselines/nel/decode.sh asr dev
```

<u>Step 2: Hyperparameter search on dev.</u>
```
bash baselines/nel/eval_nel.sh ppl
```

#### pipeline-oracle
Perfect ASR: assuming access to GT transcripts, so the predicted time stamps are same as the GT force-aligned time-stamps. 

<u>Evaluate output of the [text NER model](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/ner#fine-tuning-nlp-topline) on dev.</u>
```
bash baselines/nel/eval_nel.sh oracle_ppl
```