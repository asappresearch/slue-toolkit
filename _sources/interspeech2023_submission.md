# Submissions
All participants should upload their system description (template link) and output TSV files. Participants are allowed to submit total 3 results (3 TSV files) per track.
Please follow the submission format to be evaluated correctly. The formats are extactly the same that you use for the SLUE-Toolkit.

## Submission format for pipeline track

in .tsv format, prepare 3 columns (`id`, `pred_text`, `pred_ner`) as in the below example. Please specify none, if the prediction does not exist.

````
id	pred_text	pred_ner
20150518-0900-PLENARY-15-en_20150518-18:48:27_2	we all agreed at the last session in strasbourg that development is important but we need to remember it now when we are talking about the financial contributions	[['GPE', 'strasbourg']]
20180530-0900-PLENARY-26-en_20180530-20:01:52_0	madam president first of all i would like to thank all the members who have participated in this important debate for their different contributions. None

````

## Submission format for E2E track
in .tsv format, prepare 2 columns (`id`, `pred_ner`) as in the below example. Please specify none, if the prediction does not exist.

````
id	pred_ner
20150518-0900-PLENARY-15-en_20150518-18:48:27_2	[['GPE', 'strasbourg']]
20180530-0900-PLENARY-26-en_20180530-20:01:52_0	 None

````

## Submission

Submission URL : TBA
