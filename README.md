# BERT-BC
BERT-BC: A Unified Alignment and Interaction Model over Hierarchical BERT for Response Selection

- Unzip the data to the designated directory, Data_processing.py data_preprocess.py for converting data to json format, fine_turning_preprocess.py for processing fine-tuning data, tokenieze_post_train.py for processing post-train dat.
Different data sets can be handled by modifying the data_type parameter.
- Data_processing.py 
- fine_turning_preprocess.py
- tokenize_post_train.py

# dataset 
Ubuntu dataset: https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip

Douban dataset: https://www.dropbox.com/s/90t0qtji9ow20ca/DoubanConversaionCorpus.zip?dl=0

E Commerce dataset: https://drive.google.com/file/d/154J-neBo20ABtSmJDvm7DK0eTuieAuvw/view

preprocessed data: https://github.com/hanjanghoon/BERT_FP

# post-pretrain
Download initial checkpoint from https://huggingface.co/models
```
python -u pretrain_final.py --model_class du_bert_pretrain --batch_size 32 --task e_commerce --is_training --epochs 15
```
```
python -u pretrain_final.py --model_class du_bert_pretrain --batch_size 32 --task douban --is_training --epochs 13
```
```
python -u pretrain_final.py --model_class du_bert_pretrain --batch_size 32 --task ubuntu --is_training --epochs 8
```

# fine-turning

```
python -u bert_fineturning_cul.py --model_class du_bert_pretrain --batch_size 32 --task e_commerce --learning_rate 1e-6  --epochs 10 --is_training
```
```
python -u bert_fineturning_cul.py --model_class du_bert_pretrain --batch_size 32 --task douban --learning_rate 2e-7 --epochs 5 --is_training
```
```
python -u bert_fineturning_cul.py --model_class du_bert_pretrain --batch_size 32 --task ubuntu --learning_rate 1e-6  --epochs 3 --is_training
```
