The `dev/`, `finetune/`, and `test/` folders contain the audio files from the corresponding split. 

`slue-voxpopuli_fine-tune.tsv` and `slue-voxpopuli_dev.tsv` have the text transcripts and NER tags corresponding to these audio files. Note that `slue-voxpopuli_test_blind.tsv` includes only the text transcripts and not the NER tags.

The NER tags are formated as a list of lists, where each constitute list has three elements: `[NER tag, start character index of NER phrase in the transcript, number of characters in the NER phrase]` 
For example: 
```
id      raw_text        normalized_text speaker_id      split   raw_ner normalized_ner
20131007-0900-PLENARY-19-en_20131007-21:26:04_7         this could be done by looking more specifically into the suggested fallback clause in article twenty one in cases brought by the employee against the employer defining as relevant the place of business from which the employee receives or received day to day instructions  None    fine-tune       []      [['LAW', 86, 18]]
```
The entity phrase `article twenty one` (length 18) has NER label `LAW`. The corresponding audio file can be found at `fine-tune/20131007-0900-PLENARY-19-en_20131007-21:26:04_7.ogg`.

