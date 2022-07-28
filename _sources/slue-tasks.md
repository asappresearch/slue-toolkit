# SLUE Tasks
## Tasks

Task | Primary metric
---|---
Automatic Speech Recognition (ASR)| WER
Named Entity Recognition (NER)| F1
Sentiment Analysis | F1

## Datasets

SLUE uses the [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and [VoxPopuli](https://github.com/facebookresearch/voxpopuli) datasets. We carefully curated subsets of these datasets for fine-tuning and evaluation for SLUE tasks, and we re-distribute the subsets. Thus, you don’t need to download the whole (gigantic) datasets. In the dataset, we also include the human annotation and transcription for SLUE tasks. All you need to do is run the script below and it will download and pre-process the dataset.

Here is a brief info about the datasets.  Additional details are in the [paper](https://arxiv.org/pdf/2111.10367.pdf).

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
    <td>1,454 (3.2)</td>
    <td>3,553 (7.8)</td>
    <td>ASR, SA</td>
    <td>CC-BY 4.0 (check complete license <a href="https://papers-slue.awsdev.asapp.com/slue-voxceleb_LICENSE">here</a>)</td>
<!--     <td>CC-BY 4.0</td>
    <td>CC-BY 4.0</td> -->
  </tr>
</tbody>
</table>
