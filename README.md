# slue-toolkit
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

We introduce Spoken Language Understanding Evaluation (SLUE) benchmark. This toolkit provides codes to download and pre-process the SLUE datasets, train the baseline models, and evaluate SLUE tasks. Refer [https://arxiv.org/abs/2111.10367](https://arxiv.org/abs/2111.10367) for more details.

## News
 - Jul. 28, 2022: We update the data to v0.2 where the dev set of slue-voxceleb has a similar distribution as the test set. 
 - Nov. 22, 2021: We release the SLUE paper on arXiv along with the slue-toolkit repository. The repository contains data processing and evaluation scripts. We will publish the scripts for training the baseline models soon.

## Installation
1. git clone this repository and install slue-toolkit (development mode)
```sh
git clone https://github.com/asappresearch/slue-toolkit.git
pip install -e .
```
or install directly from Github
```sh
pip install git+https://github.com/asappresearch/slue-toolkit.git
```
2. Install additional dependency based on your choice (e.g. you need `fairseq` and ``transformers`` for baselines)

## SLUE Tasks
### Automatic Speech Recognition (ASR)

Although this is not a SLU task, ASR can help analyze the performance of downstream SLU tasks on the same domain. Additionally, pipeline approaches depend on ASR outputs, making ASR relevant to SLU. ASR is evaluated using word error rate (WER).

### Named Entity Recognition (NER)

Named entity recognition involves detecting the named entities and their tags (types) in a given sentence. We evaluate performance using micro-averaged F1 and label-F1 scores. The F1 score evaluates an unordered list of named entity phrase and tag pairs predicted for each sentence. Only the tag predictions are considered for label-F1.

### Sentiment Analysis (SA)

Sentiment analysis refers to classifying a given speech segment as having negative, neutral, or positive sentiment. We evaluate SA using  macro-averaged (unweighted) recall and F1 scores.

### Datasets

<table>
<thead>
  <tr>
    <th rowspan="2">Corpus</th>
    <th colspan="3">Size - utts (hours)</th>
    <th rowspan="2">Tasks</th>
    <th rowspan="2">License</th>
  </tr>
  <tr>
    <th>Fine-tune</th>
    <th>Dev</th>
    <th>Test</th>
<!--     <th>Audio</th>
    <th>Text</th>
    <th>Annotation</th> -->
  </tr>
</thead>
<tbody>
  <tr>
    <td>SLUE-VoxPopuli</td>
    <td>5,000 (14.5)</td>
    <td>1,753 (5.0)</td>
    <td>1,842 (4.9)</td>
    <td>ASR, NER</td>
   <td>CC0 (check complete license <a href="https://papers-slue.awsdev.asapp.com/slue-voxpopuli_LICENSE">here</a>)</td>
<!--     <td>CC0</td>
    <td>CC0</td> -->
  </tr>
  <tr>
    <td>SLUE-VoxCeleb</td>
    <td>5,777 (12.8)</td>
    <td>955 (2.1)</td>
    <td>4,052 (9.0)</td>
    <td>ASR, SA</td>
    <td>CC-BY 4.0 (check complete license <a href="https://papers-slue.awsdev.asapp.com/slue-voxceleb_LICENSE">here</a>)</td>
<!--     <td>CC-BY 4.0</td>
    <td>CC-BY 4.0</td> -->
  </tr>
</tbody>
</table>

For SLUE, you need [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and [VoxPopuli](https://github.com/facebookresearch/voxpopuli) dataset. We carefully curated subset of those dataset for fine-tuning and evaluation for SLUE tasks, and we re-distribute the the subsets. Thus, you don't need to download a whole gigantic datasets. In the dataset, we also includes the human annotation and transcription for SLUE tasks. All you need to do is just running the script below and it will download and pre-process the dataset.

#### Download and pre-process dataset

```sh
bash scripts/download_datasets.sh
```


## SLUE score evaluation
The test set data and annotation will be used for the official SLUE score evaluation, however we will not release the test set annotation. Thus, the SLUE score can be evaluated by submitting your prediction result in tsv format. We will prepare the website to accept your submission. Please stay tuned for this.

## Model development rule
To train model, You can use fine-tuning and dev sets (audio, transcription and annotation) except the test set of SLUE task. Additionally you can use any kind of external dataset whether it is labeled or unlabeled for any purpose of training (e.g. pre-training and fine-tuning).

For vadidation of your model, you can use official dev set we provide, or you can make your own splits or cross-validation splits by mixing fine-tuning and dev set all together.

## Baselines

### ASR
#### Fine-tuning
Assuming that the preprocessed manifest files are in `manifest/slue-voxceleb` and `manifest/slue-voxpopuli` for SLUE-VoxCeleb and SLUE-VoxPopuli. This command fine-tune a wav2vec 2.0 base model on these two datasets using one GPU.
```sh
bash baselines/asr/ft-w2v2-base.sh manifest/slue-voxceleb save/asr/w2v2-base-vc
bash baselines/asr/ft-w2v2-base.sh manifest/slue-voxpopuli save/asr/w2v2-base-vp
```

#### Evaluation
To evaluate the fine-tuned wav2vec 2.0 ASR models on the dev set, please run the following commands.
```sh
python slue_toolkit/eval/eval_w2v.py eval_asr save/asr/w2v2-base-vc --data manifest/slue-voxceleb --subset dev
python slue_toolkit/eval/eval_w2v.py eval_asr save/asr/w2v2-base-vp --data manifest/slue-voxpopuli --subset dev
```
The WER will be printed directly.
The predictions are saved in `save/asr/w2v2-base-vc/pred-dev.wrd` and `save/asr/w2v2-base-vp/pred-dev.wrd` and can be used for pipeline models.

More detail baseline experiment described [here](baselines/asr/README.md)

### NER
#### Fine-tuning End-to-end model
Assuming that the preprocessed manifest files are in `manifest/slue-voxpopuli` for SLUE-VoxPopuli. This command fine-tune a wav2vec 2.0 base model using one GPU.
```sh
bash baselines/ner/e2e_scripts/ft-w2v2-base.sh manifest/slue-voxpopuli/e2e_ner save/e2e_ner/w2v2-base
```

#### Evaluating End-to-End model

To evaluate the fine-tuned wav2vec 2.0 E2E NER model on the dev set, please run the following command. (decoding without language model)
```sh
bash baselines/ner/e2e_scripts/eval-ner.sh w2v2-base dev combined nolm
```
More detail baseline experiment described [here](baselines/ner/README.md)


### Sentiment Analysis
#### Fine-tuning
This command fine-tune a wav2vec 2.0 base model on the voxceleb dataset
```sh
bash baselines/sentiment/e2e_scripts/ft-w2v2-base-senti.sh manifest/slue-voxceleb save/sentiment/w2v2-base
```
#### Evaluation
To evaluate the fine-tuned wav2vec 2.0 sentiment model, run following commands or run `baselines/sentiment/e2e_scripts/eval.sh`
```sh
python3 slue_toolkit/eval/eval_w2v_sentiment.py --save-dir save/sentiment/w2v2-base --data manifest/slue-voxceleb --subset dev
```
More detail baseline experiment described [here](baselines/sentiment/README.md)

# How-to-submit for your test set evaluation

See here https://asappresearch.github.io/slue-toolkit/how-to-submit.html

