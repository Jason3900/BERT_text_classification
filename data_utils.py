# -*- coding:UTF-8 -*-
"""
# @Time: 2020-11-22
# @Author: 方雪至
# @Affiliation: 北京语言大学语言监测与智能学习研究小组
# @email: jasonfang3900@gmail.com
"""
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import pickle


class TextClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def ssplit(text):
    text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)
    text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)
    text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
    text = re.sub("\n+", "\n", text)
    return text

def count_labels(labels):
    label_count_dict = dict()
    for label in labels:
        if label not in label_count_dict:
            label_count_dict[label] = 0
        label_count_dict[label] += 1
    label_count_result = sorted(label_count_dict.items(), key=lambda x: x[0], reverse=False)
    return label_count_result

def read_data(filepath):
    docs = []
    labels = []
    with open(filepath, "r", encoding="utf8") as fr:
        for line in fr:
            doc, label = line.split("\t")
            docs.append(doc)
            labels.append(int(label))
    return docs, labels


def split_dataset(docs, labels, random_seed) -> dict:
    train_data, dev_data, train_labels, dev_labels = train_test_split(docs,labels,test_size=0.2, stratify=labels, random_state=random_seed)
    # dev_data, test_data, dev_labels, test_labels = train_test_split(val_data,val_labels,test_size=0.5, stratify=val_labels)
    dataset = dict(
        train_dataset = [train_data, train_labels], 
        dev_dataset = [dev_data, dev_labels],
        # test_dataset = [test_data, test_labels]
    )
    train_label_count = count_labels(train_labels)
    dev_label_count = count_labels(dev_labels)
    # test_label_count = count_labels(test_labels)
    print(f"train_data: {len(train_labels)} samples.")
    print(f"train label count: {train_label_count}")
    print(f"dev_data: {len(dev_labels)} samples.")
    print(f"dev label count: {dev_label_count}")
    # print(f"test_data: {len(test_labels)} samples.")
    # print(f"test label count: {test_label_count}")
    return dataset


