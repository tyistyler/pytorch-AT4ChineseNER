import os

# export CUDA_VISIBLE_DEVICES=4
# dataset name
dataset = "weibo"
attn_type = "dot"
# Path of bert model
bert_model = "data/bert-base-chinese"
pool_method = "first"
# Path of the ZEN model
zen_model = "./data/ZEN_pretrain_base"


log = "log/{}_bert_{}.txt".format(dataset, pool_method)

#       num_layers = 1
os.system("python3 train_zen_cn.py --ner_dataset weibo "
          "--seed 14 --bert_model {} --pool_method first "
          "--lr 0.001 --fc_dropout 0.3 "
          "--log {} --batch_size 16 ".format(bert_model, log))
          