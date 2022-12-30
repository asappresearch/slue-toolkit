# SLUE 2022 Challange

## NER Task definition
Named entity recognition involves detecting the named entities and their tags (types) in a given sentence. Named entities are phrases, often (but not always) consisting of proper nouns, that refer to distinct entities such as a person, location, organization, numerical value, etc.

For NER labels, we use `combined label` set as defined in [SLUE](https://arxiv.org/abs/2111.10367) phase-1 paper. We evaluate performance using micro-averaged F1 scores on two aspects of the output. F1 score is the harmonic mean of precision and recall. The first score, referred to as F1, evaluates an unordered list of named entity phrase and tag pairs predicted for each sentence. The second score, label-F1, considers only the tag predictions. Label-F1 is useful to understand model accuracy despite the possible misspelling and segmentation errors in speech-to-text conversion.

## Challenge tracks and rules
A participant must report at least one of the system's output either pipeline or E2E. Pipeline models generate transcription from speech using an ASR system, and then apply an NLP model on the generated ASR transcription. End-to-end (E2E) models directly map from the input speech to output task labels. Participants can use the audio, transcription and NER labels in `fine-tune` and `dev` set of SLUE-voxpopuli dataset for any purpose, such as training, fine-tuning, validation or calibration. Participants should not use the `test` set for any kind of purpose. No label will be provided for the test dataset. All performance measurements will be finally made on the test set. Test set label will be available after result submission to help participants analyze their result and report their findings.

#### General rules
Note that SLUE-Voxpopuli is a subset of the Voxpopuli dataset and participants have the freedom to use the entire Voxpopuli dataset. However, it is not allowed an additional human annotation for the named entity on the Voxpopuli dataset. Participants are only allowed to use human annotation already provided in SLUE-Voxpopuli.

#### pipeline track
Participants are free to use any kind open-sourced pre-trained model for pipeline system. The acoustic pre-trained model can be supervised model for ASR or unsupervised pre-trained model to be fine-tuned using any kind of speech data with or without ground truth transcription including SLUE-Voxpopuli. The NLP pre-trained model is also can be any kind of model that trained unsupervisedly or supervised way for any kind of NLP tasks.

#### E2E track
Participants are free to use any kind of open-sourced pre-trained model. Here, E2E system means that the model can directly map the audio to NER output using acoustic model. However, participants could still use NLP model or another models to train E2E system for example, distillation or transfer learning task.

## SLUE-Voxpopuli dataset

Please find [SLUE](https://arxiv.org/abs/2111.10367) phase-1 paper for SLUE-Voxpopuli dataset detail. You can download dataset using [SLUE-Toolkit](https://github.com/asappresearch/slue-toolkit)

## Toolkit and baseline

[SLUE-Toolkit](https://github.com/asappresearch/slue-toolkit) provide all related scripts such as downloading dataset, pre-processing dataset, preparing combined label, and baseline reciepes. A detail baseline for NER is [here](https://github.com/asappresearch/slue-toolkit/tree/main/baselines/ner).
