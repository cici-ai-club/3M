# 3M_stylish
Example script for Data Processing
1. prepare labels
```
python scripts/prepro_labels.py 
--input_json ParlAI/data/personality_captions/dataset_person.json 
--output_json data/personcap_added1.json 
--output_h5 data/personcap_added1  
--perssOnehot_h5 data/person_onehot_added1 
--densecap_h5 data/densecap_added1 
--personality_path ParlAI/data/personality_captions/personalities.txt
```
2. prepare ngrams
```
python scripts/prepro_ngrams.py 
--rm_punc 0 
--input_json ParlAI/data/personality_captions/dataset_person.json 
--dict_json data/personcap_added1.json 
--split val 
--output_pkl data/person-val
```
split above could be changed to test, train or all.
3. prepare reference for evaluation
```
python scripts/prepro_reference_json.py 
--rm_punc 0 
--input_json ParlAI/data/personality_captions/dataset_person.json 
--dict_json data/personcap_added1.json 
--output_json coco-caption/person_captions4eval_-1.json 
--gdindex -1
```
gdindex and the output_json file name could be changed from (0-4)


