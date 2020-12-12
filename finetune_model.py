# -*- coding:UTF-8 -*-
"""
# @Time: 2020-11-22
# @Author: 方雪至
# @Affiliation: 北京语言大学语言监测与智能学习研究小组
# @email: jasonfang3900@gmail.com
"""
from transformers import BertModel
import torch.nn as nn

class BertClassificationModel(nn.Module):

    def __init__(self, config):
        super(BertClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = config.requires_grad
        # self.dropout = nn.Dropout(p=config.dropout)
        self.linear = nn.Linear(config.linear_hidden_size, config.num_labels)

    def forward(self, data):
        _, pooled = self.bert(**data, output_hidden_states=False)
        # drop_pooled = self.dropout(pooled)
        out = self.linear(pooled)
        return out