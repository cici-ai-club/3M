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
split above could be changed to test, train or all. <br />
3. prepare reference for evaluation
```
python scripts/prepro_reference_json.py 
--rm_punc 0 
--input_json ParlAI/data/personality_captions/dataset_person.json 
--dict_json data/personcap_added1.json 
--output_json coco-caption/person_captions4eval_-1.json 
--gdindex -1
```
gdindex and the output_json file name could be changed from (0-4) <br />
Train the model
```
id="densepembed2_added"
python densetrain3m.py --id $id \
    --caption_model densepembed \
    --decoder_type LSTM \
    --mean_feats 1 \
    --ctx_drop 1 \
    --label_smoothing 0 \
    --input_json data/personcap_added1.json \
    --input_label_h5 data/personcap_added1_label.h5 \
    --input_fc_dir   ParlAI/data/yfcc_images/resnext101_32x48d_wsl \
    --input_att_dir  ParlAI/data/yfcc_images/resnext101_32x48d_wsl_spatial_att \
    --input_box_dir  data/cocobu_box \
    --perss_onehot_h5  data/person_onehot_added1.h5 \
    --cached_tokens  data/person-train-idxs \
    --seq_per_img 1 \
    --batch_size 128 \
    --seq_per_img 1 \
    --beam_size 1 \
    --learning_rate 5e-4 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 2048 \
    --att_hid_size 512 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --checkpoint_path log_added_new1/log_$id  \
    $start_from \
    --save_checkpoint_every 3000 \
    --language_eval 1 \
    --val_images_use -1 \
    --max_epochs 30 \
    --scheduled_sampling_increase_every 5 \
    --scheduled_sampling_max_prob 0.5 \
    --learning_rate_decay_every 5 \

```
Eval the model
```
id="densepembed2_added" 
python  denseeval3m.py --id $id \
    --dump_images 0 \ 
    --num_images -1 \
    --split test    \   
    --input_json data/personcap_added1.json \
    --input_label_h5 data/personcap_added1_label.h5 \
    --input_fc_dir   ParlAI/data/yfcc_images/resnext101_32x48d_wsl \
    --input_att_dir   ParlAI/data/yfcc_images/resnext101_32x48d_wsl_spatial_att \
    --perss_onehot_h5  data/person_onehot_added1.h5 \
    --batch_size 128 \
    --seq_per_img  5 \ 
    --beam_size 5 \ 
    --language_eval 1 \ 
    --infos_path log_added_new/log_$id/infos_$id-best.pkl \
    --model log_added_new/log_$id/model-best.pth       \   
    --temperature 1.0  
```
