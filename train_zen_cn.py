from models.TENER import SAModel
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler, SequentialSampler
from fastNLP.embeddings import StaticEmbedding, BertEmbedding, StackEmbedding
from modules.pipe import CNNERPipe
from fastNLP.io.pipe.cws import CWSPipe
from fastNLP.core.losses import LossInForward

from run_token_level_classification import BertTokenizer, ZenNgramDict, ZenForTokenClassification, load_examples
from utils_token_level_task import PeopledailyProcessor, CwsmsraProcessor

import os
import argparse
from modules.callbacks import EvaluateCallback

from datetime import datetime

import torch
import random
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--ner_dataset', type=str, default='weibo', choices=['weibo', 'resume', 'ontonotes', 'msra'])
parser.add_argument('--cws_dataset', type=str, default='msr', choices=['msr', 'pku'])
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--bert_model', type=str, required=True)
parser.add_argument('--pool_method', type=str, default="first", choices=["first", "last", "avg", "max"])
parser.add_argument('--fc_dropout', type=float, default=0.4)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_heads', type=int, default=12)
parser.add_argument('--head_dims', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
#---------------new add---------------------------
parser.add_argument("--use_memory", action='store_true', help="Whether to add memory.")
#---------------new add end---------------------------
args = parser.parse_args()

ner_dataset = args.ner_dataset
cws_dataset = args.cws_dataset
lr = args.lr
attn_type = 'adatrans'
n_epochs = 50

pos_embed = None
n_heads = args.n_heads
num_layers = args.num_layers
head_dims = args.head_dims

batch_size = args.batch_size
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True

# dropout=0.15
fc_dropout=args.fc_dropout

# new_add

encoding_type = 'bioes'
ner_name = 'caches/{}_{}_{}_{}.pkl'.format(ner_dataset, model_type, encoding_type, normalize_embed)
cws_name = 'caches/{}_{}_{}_{}.pkl'.format(cws_dataset, model_type, encoding_type, normalize_embed)
d_model = 256
dim_feedforward = int(2 * d_model)

def print_time():
    now = datetime.now()
    return "-".join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])

save_path = "ckpt/bert_{}_{}_{}.pth".format(ner_dataset, model_type, print_time())

logPath = args.log

def write_log(sent):
    with open(logPath, "a+", encoding="utf-8") as f:
        f.write(sent)
        f.write("\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
setup_seed(args.seed)
    
@cache_results(ner_name, _refresh=False)
def load_ner_data():
    ner_paths = {'train': 'data/{}/train.txt'.format(ner_dataset),
             'dev':'data/{}/dev.txt'.format(ner_dataset),
             'test':'data/{}/test.txt'.format(ner_dataset)}
    min_freq = 2
    ner_data_bundle = CNNERPipe(bigrams=True, encoding_type=encoding_type).process_from_file(ner_paths)

    # train_list = data_bundle.get_dataset('train')['raw_chars']

    # embed = StaticEmbedding(data_bundle.get_vocab('chars'),
    #                         model_dir_or_name='data/gigaword_chn.all.a2b.uni.ite50.vec',
    #                         min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    ner_bi_embed = StaticEmbedding(ner_data_bundle.get_vocab('bigrams'),
                               model_dir_or_name='data/gigaword_chn.all.a2b.bi.ite50.vec',
                               word_dropout=0.02, dropout=0.3, min_freq=2,
                               only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    ner_tencent_embed = StaticEmbedding(ner_data_bundle.get_vocab('chars'),
                                    model_dir_or_name='data/tencent_unigram.txt',
                                    min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01,
                                    dropout=0.3)

    ner_bert_embed = BertEmbedding(vocab=ner_data_bundle.get_vocab('chars'), model_dir_or_name=args.bert_model, layers='-1',
                               pool_method=args.pool_method, word_dropout=0, dropout=0.5, include_cls_sep=False,
                               pooled_cls=True, requires_grad=False, auto_truncate=True)

    ner_embed = StackEmbedding([ner_tencent_embed, ner_bert_embed], dropout=0, word_dropout=0.02)

    return ner_data_bundle, ner_embed, ner_bi_embed


@cache_results(cws_name, _refresh=False)
def load_cws_data():
    cws_paths = {'train': 'data/{}/train.txt'.format(cws_dataset),
                 'test': 'data/{}/test.txt'.format(cws_dataset)}
    min_freq = 2
    cws_data_bundle = CNNERPipe(bigrams=True, encoding_type='bmeso').process_from_file(cws_paths)

    # train_list = data_bundle.get_dataset('train')['raw_chars']

    # embed = StaticEmbedding(data_bundle.get_vocab('chars'),
    #                         model_dir_or_name='data/gigaword_chn.all.a2b.uni.ite50.vec',
    #                         min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01, dropout=0.3)

    cws_bi_embed = StaticEmbedding(cws_data_bundle.get_vocab('bigrams'),
                                   model_dir_or_name='data/gigaword_chn.all.a2b.bi.ite50.vec',
                                   word_dropout=0.02, dropout=0.3, min_freq=2,
                                   only_norm_found_vector=normalize_embed, only_train_min_freq=True)

    cws_tencent_embed = StaticEmbedding(cws_data_bundle.get_vocab('chars'),
                                        model_dir_or_name='data/tencent_unigram.txt',
                                        min_freq=1, only_norm_found_vector=normalize_embed, word_dropout=0.01,
                                        dropout=0.3)

    cws_bert_embed = BertEmbedding(vocab=cws_data_bundle.get_vocab('chars'), model_dir_or_name=args.bert_model,
                                   layers='-1',
                                   pool_method=args.pool_method, word_dropout=0, dropout=0.5, include_cls_sep=False,
                                   pooled_cls=True, requires_grad=False, auto_truncate=True)

    cws_embed = StackEmbedding([cws_tencent_embed, cws_bert_embed], dropout=0, word_dropout=0.02)

    return cws_data_bundle, cws_embed, cws_bi_embed

ner_data_bundle, ner_embed, ner_bi_embed = load_ner_data()
cws_data_bundle, cws_embed, cws_bi_embed = load_cws_data()

ner_bi_embed=None
cws_bi_embed=None

print(ner_data_bundle)
print(cws_data_bundle)

task_ner = torch.LongTensor([1] * batch_size).to(device)
task_cws = torch.LongTensor([0] * batch_size).to(device)

model = SAModel(ner_tag_vocab=ner_data_bundle.get_vocab('target'),cws_tag_vocab=cws_data_bundle.get_vocab('target'),
                ner_embed=ner_embed, cws_embed=cws_embed, num_layers=num_layers,
                d_model=d_model, n_head=n_heads,
                feedforward_dim=dim_feedforward, dropout=0.3,
                after_norm=after_norm, attn_type=attn_type,
                ner_bi_embed=None,cws_bi_embed=None,
                fc_dropout=fc_dropout,
                pos_embed=pos_embed,
                scale=attn_type=='naive',
                gram2id=None, device=device,
                task_ner=task_ner,task_cws=task_cws
              )

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
cws_optimizer = None
callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data=ner_data_bundle.get_dataset('test'),
                                     args=args,
                                     gram2id=None,
                                     device=device,
                                     dataset='test'
                                     )

if warmup_steps > 0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(ner_data_bundle.get_dataset('train'), cws_data_bundle.get_dataset('train'),
                  model, optimizer, cws_optimizer=None,
                  cws_losser=LossInForward(loss_key='cws_loss'),
                  joint_losser=LossInForward(loss_key='adv_loss'),
                  batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=0, n_epochs=n_epochs, dev_data=ner_data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=ner_data_bundle.get_vocab('target'), encoding_type=encoding_type),
                  dev_batch_size=batch_size, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=save_path,
                  logger_func=write_log,
                  gram2id=None, args=args
                  )

trainer.train(load_best_model=False)

