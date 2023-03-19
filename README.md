# 3M
## Architecture 
![Alt text](architecture_3m.png?raw=true "Title")
## Pretrained Features
1. Extract Dense captions features(follow the code by https://github.com/jcjohnson/densecap)
2. ResNext features (We extract them by following the instructions under https://github.com/facebookresearch/ParlAI), the features we used are mean-pooled image  features saved in resnext101_32x48d_wsl/ and spatial feature saved in resnext101_32x48d_wsl_spatial_att/. 
Note: we do not change the netwrok in dense caption or ResNext network, we just directly use the pretrained network to generate our features for stylish captioning task.
## Updates
1. The images we didn't use is the ones cannot be downloaded from https://github.com/facebookresearch/ParlAI, which are ac8*.jpg
2. I added img_caption.json (this files contains dense captions extracted from the images) into drive: https://drive.google.com/drive/folders/170palQ7QzRsY2ZRyaDTIQAcdHyuVZsFe?
3. If there is any I remember later
## Example script for Data Processing
1. prepare labels
```
python scripts/prepro_labels.py 
--input_json data/dataset_person.json 
--output_json data/personcap_added1.json 
--output_h5 data/personcap_added1  
--perssOnehot_h5 data/person_onehot_added1 
--densecap_h5 data/densecap_added1 
--personality_path ParlAI/data/personality_captions/personalities.txt
```
We convert original personality caption dataset to dataset_person.json based on the format recommended in https://github.com/ruotianluo/self-critical.pytorch <br />
personalities.txt could be downloaded through https://github.com/facebookresearch/ParlAI by <br />
2. prepare ngrams
```
python scripts/prepro_ngrams.py 
--rm_punc 0 
--input_json data/dataset_person.json 
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
## Train the model
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
    --input_fc_dir   data/yfcc_images/resnext101_32x48d_wsl \
    --input_att_dir  data/yfcc_images/resnext101_32x48d_wsl_spatial_att \
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
## Eval the model
```
id="densepembed2_added" 
python  denseeval3m.py --id $id \
    --dump_images 0 \ 
    --num_images -1 \
    --split test    \   
    --input_json data/personcap_added1.json \
    --input_label_h5 data/personcap_added1_label.h5 \
    --input_fc_dir   data/yfcc_images/resnext101_32x48d_wsl \
    --input_att_dir   data/yfcc_images/resnext101_32x48d_wsl_spatial_att \
    --perss_onehot_h5  data/person_onehot_added1.h5 \
    --batch_size 128 \
    --seq_per_img  5 \ 
    --beam_size 5 \ 
    --language_eval 1 \ 
    --infos_path log_added_new/log_$id/infos_$id-best.pkl \
    --model log_added_new/log_$id/model-best.pth       \   
    --temperature 1.0  
```
## Large files you could get from us.
Pretrained Model, extracted dense caption and reformated personality caption data could get from here:
https://drive.google.com/drive/folders/170palQ7QzRsY2ZRyaDTIQAcdHyuVZsFe?
# Reference
```
@inproceedings{johnson2016densecap,
  title={Densecap: Fully convolutional localization networks for dense captioning},
  author={Johnson, Justin and Karpathy, Andrej and Fei-Fei, Li},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4565--4574},
  year={2016}
}
@inproceedings{shuster2019engaging,
  title={Engaging image captioning via personality},
  author={Shuster, Kurt and Humeau, Samuel and Hu, Hexiang and Bordes, Antoine and Weston, Jason},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12516--12526},
  year={2019}
}
@inproceedings{anderson2018bottom,
  title={Bottom-up and top-down attention for image captioning and visual question answering},
  author={Anderson, Peter and He, Xiaodong and Buehler, Chris and Teney, Damien and Johnson, Mark and Gould, Stephen and Zhang, Lei},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6077--6086},
  year={2018}
}
```
# Citation
Cite this paper if you find anything useful or use the code.
```
@misc{2103.11186,
Author = {Chengxi Li and Brent Harrison},
Title = {3M: Multi-style image caption generation using Multi-modality features under Multi-UPDOWN model},
Year = {2021},
Eprint = {arXiv:2103.11186},
}

```
# Acknowledge 
A lot of code of here are derived from Ruotian's repo https://github.com/ruotianluo/self-critical.pytorch
