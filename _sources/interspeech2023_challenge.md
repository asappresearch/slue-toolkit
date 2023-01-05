# SLUE 2023 Challange Rule

## NER Task definition
Named entity recognition involves detecting the named entities and their tags (types) in a given sentence. Named entities are phrases, often (but not always) consisting of proper nouns, that refer to distinct entities such as a person, location, organization, numerical value, etc.

For NER labels, we use `combined label` set as defined in [SLUE](https://arxiv.org/abs/2111.10367) phase-1 paper. We evaluate performance using micro-averaged F1 score of the output. The F1 score evaluates an unordered list of named entity phrase and tag pairs predicted for each sentence.

## Challenge tracks and rules
A participant must report at least one of the system’s output, either pipeline or E2E. Pipeline models generate transcription from speech using an ASR system, and then apply an NLP model on the generated ASR transcription. End-to-end (E2E) models directly map from the input speech to output task labels. Participants can use the audio, transcription and NER labels in `fine-tune` and `dev` set of SLUE-voxpopuli dataset for any purpose, such as training, fine-tuning, validation, or calibration. Participants should not use the `test` set for any kind of purpose. No label will be provided for the test dataset. All performance measurements will be made on the test set. Test set labels will be available after result submission to help participants analyze their result and report their findings.

#### General rules
Note that SLUE-Voxpopuli is a subset of the Voxpopuli dataset and participants have the freedom to use the entire Voxpopuli dataset. However, it is not allowed an additional human annotation for the named entity on the Voxpopuli dataset. Participants are only allowed to use human annotation already provided in SLUE-Voxpopuli.

#### pipeline track
Participants are free to use any kind of open-sourced pre-trained model for the pipeline system. The acoustic pre-trained model can be a supervised model for ASR or unsupervised pre-trained model to be fine-tuned using any kind of speech data with or without ground truth transcription, including SLUE-Voxpopuli. The NLP pre-trained model can be any kind of model that's trained, unsupervised or supervised for any kind of NLP tasks.

#### E2E track
Participants are free to use any kind of open-sourced pre-trained model. Here, E2E system means that the model can directly map the audio to the NER output using an acoustic model. However, participants could still use the NLP model or other models to train the E2E system, for example, distillation or transfer learning tasks.

## SLUE-Voxpopuli dataset
Please find [SLUE](https://arxiv.org/abs/2111.10367) phase-1 paper for the SLUE-Voxpopuli dataset detail. You can download the dataset using [SLUE-Toolkit](https://github.com/asappresearch/slue-toolkit)

## Toolkit and baseline

[SLUE-Toolkit](https://github.com/asappresearch/slue-toolkit) provides all related scripts: the downloadable dataset, pre-processing dataset, preparing combined label, and baseline recipes. A detailed baseline for NER can be found [here](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/ner).
