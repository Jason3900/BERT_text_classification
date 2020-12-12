# -*- coding:UTF-8 -*-
"""
# @Time: 2020-11-22
# @Author: 方雪至
# @Affiliation: 北京语言大学语言监测与智能学习研究小组
# @email: jasonfang3900@gmail.com
"""
import torch

class Config():
    def __init__(self):
        self.bert_path = "./bert-base-chinese"
        self.data_path = "./train_all.txt"
        self.max_len = 512
        self.random_seed = 2
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_batch_size = 128
        self.eval_batch_size = 32
        self.epochs = 1
        self.lr = 4e-5
        self.num_labels = 10
        self.dropout = 0
        self.requires_grad = False
        self.saved_prefix = "./saved_models/BertForSeqClassify"
        self.linear_hidden_size = 768

