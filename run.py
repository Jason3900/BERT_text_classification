# -*- coding:UTF-8 -*-
"""
# @Time: 2020-11-22
# @Author: 方雪至
# @Affiliation: 北京语言大学语言监测与智能学习研究小组
# @email: jasonfang3900@gmail.com
"""
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from finetune_model import BertClassificationModel
import torch
import numpy as np
from tqdm import tqdm
from train_and_eval import train_fn, eval_fn, evaluate
from data_utils import *
from param_config import Config
# import wandb
import warnings


def main():
    warnings.filterwarnings("ignore")
    # wandb.init(project="l2_classification with BertForSeqClassify")
    config = Config()
    
    # set random seed
    random_seed = config.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    

    print("load data ...")
    train_data, train_labels = read_data(config.data_path)
    print(len(train_data), len(train_labels))
    print("split dataset ...")
    data = split_dataset(train_data, train_labels, random_seed)

    print("load tokenizer ...")
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, tokenize_chinese_chars=True)
    print("encode input sents ...")
    train_encodings = tokenizer(data["train_dataset"][0], truncation=True, padding=True, return_tensors="pt", max_length=config.max_len)
    dev_encodings = tokenizer(data["dev_dataset"][0], truncation=True, padding=True, return_tensors="pt",max_length=config.max_len)
    # test_encodings = tokenizer(data["test_dataset"][0], truncation=True, padding=True, return_tensors="pt", max_length=config.max_len)


    print("split dataset ...")
    train_dataset = TextClassifyDataset(train_encodings, data["train_dataset"][1])
    dev_dataset = TextClassifyDataset(dev_encodings, data["dev_dataset"][1])
    # test_dataset = TextClassifyDataset(test_encodings, data["test_dataset"][1])

    print("create dataloader ...")
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.eval_batch_size, shuffle=True)

    model = BertClassificationModel(config)
    model = torch.nn.parallel.DataParallel(model)
    model.to(config.device)

    # optimizer settings
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)

    num_training_steps = int(len(train_dataset) / (config.epochs * config.train_batch_size))

    # training
    best_acc = 0
    dev_accs_list = []
    print("start training ...")
    for epoch in range(config.epochs):
        save_path = config.saved_prefix+f"_epoch{epoch+1}.pt"
        train_loss = train_fn(train_loader, model, optimizer, config.device)
        torch.save(model.module.state_dict(), save_path)
        dev_acc, class_report, conf_matrix, dev_loss = eval_fn(dev_loader, model, config.device, save_path, config.num_labels)
        dev_accs_list.append(dev_acc)
        if dev_acc > best_acc:
            best_acc = dev_acc
        print(f"epoch {epoch+1} training finished!")
        print(f"train_loss: {train_loss}")
        print(f"dev_acc: {dev_acc}, dev_loss: {dev_loss}")
        print(f"best performance on dev set appears in epoch {dev_accs_list.index(best_acc)+1}! ")
        print(class_report)
        print(conf_matrix)
        # wandb.log(dict(train_loss=train_loss, dev_acc=dev_acc, dev_loss=dev_loss, best_dev_acc=best_acc))


    

if __name__ == "__main__":
    main()

