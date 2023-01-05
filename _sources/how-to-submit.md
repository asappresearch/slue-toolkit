# How-to-submit

SLUE test set evaluation isn't publicly available to ensure a fair evaluation. Kindly submit the form, and our team will review it. Your submission will remain confidential.


## Submission format

Prepare a .tsv file with 4 columns as demonstrated in the illustration below. Please specify none, if the prediction does not exist. For example, Voxpopuli is for ASR and NER, it needs `id`, `pred_text`, `pred_ner` columns, and all entities in `pred_sentiment` should be `None`. If no NE found, `pred_ner` should be `None`


````
id	pred_text	pred_ner	pred_sentiment
id10012_0AXjxNXiEzo_00001	like i said less manicured in a good way i think i think that what you know people	None	Positive
20150518-0900-PLENARY-15-en_20150518-18:48:27_2	we all agreed at the last session in strasbourg that development is important but we need to remember it now when we are talking about the financial contributions	[['GPE', 'strasbourg']]	None
20180530-0900-PLENARY-26-en_20180530-20:01:52_0	madam president first of all i would like to thank all the members who have participated in this important debate for their different contributions. None None

````


## Missing Tasks

Your submission must contain at least one of the SLU tasks other than ASR to be evaluated. We'll still accept submissions with or without `pred_text` if one of `pred_ner` or `pred_sentiment` exists, but we will not evaluate your submission if you only submit `pred_text`.

However, for example, if you only want to evaluate the sentiment analysis system, you could submit only the prediction result of the sentiment analysis system (`pred_sentiment`).

Most importantly, if you fail to include any of the columns in your submission, we won't be able to calculate your SLUE score and thus cannot rank it.

## Submission

Send tsv file to slue-committee "AT" googlegroups.com
