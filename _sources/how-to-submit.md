# How-to-submit

SLUE test set evaluation is not publicly available for a fair evaluation. If you submit your test set evaluation following the desired form, we will evaluate your system.

## Submission format

in .tsv format, prepare 4 columns as in the below example. Please specify none, if the prediction does not exist (for example, Voxpopuli is for ASR and NER, it needs `id`, `pred_text`, `pred_ner` columns, and all entities in `pred_sentiment` should be `none`

````
id	pred_text	pred_ner	pred_sentiment
id10012_0AXjxNXiEzo_00001	like i said less manicured in a good way i think i think that what you know people	none	Positive
20150518-0900-PLENARY-15-en_20150518-18:48:27_2	we all agreed at the last session in strasbourg that development is important but we need to remember it now when we are talking about the financial contributions	[['GPE', 37, 10]]	none

````

## Submission

Send tsv file to sshon@asapp.com.
