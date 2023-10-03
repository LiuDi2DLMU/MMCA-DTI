# -*- coding: utf-8 -*-
import csv
import os

import dgl
import numpy as np
import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader

import models
from frame.Dataset import DTIDataset
from frame.Trainer import Trainer
from hyperparameter import Hyperparameter

from raguler import PolyLoss


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:"

def collate_fn(data):
    """
    用来将数据转化成mini-batch大小的张量
    :param data:
    :return:
    """
    return_drug_info = Hyperparameter().return_drug_info
    drug = [[] for _ in range(len(return_drug_info))]
    target = []
    labels = []
    for i in range(len(data)):
        for x in range(len(return_drug_info)):
            drug[x].append(data[i][0][x])
        target.append(data[i][1])
        labels.append(data[i][2])
    for i, info in enumerate(return_drug_info):
        if info == "molGraph":
            drug[i] = dgl.batch(drug[i])
        else:
            drug[i] = torch.tensor(np.array(drug[i]))
    target = torch.tensor(np.array(target))
    labels = torch.tensor(labels, dtype=torch.long)
    return tuple(drug), target, labels


def main():
    RDLogger.DisableLog('rdApp.*')
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    k_fold = 5

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default="DrugBank", help="The dataset used for training")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--load_weight', type=str, default=None, help="Whether to load pre-training weights")
    parser.add_argument("--e", type=int, default=1, help="Which dataset settings to load")

    args = parser.parse_args()
    hp = Hyperparameter()

    hp.kfold = k_fold
    hp.load_weight = args.load_weight
    # 模型
    hp.model = models.MMCADTI
    # 所用数据集
    hp.dataset = args.dataset
    hp.dataModel = "E" + str(args.e)

    if hp.dataset == "KIBA":
        # hp.weight = [1, 4.25]
        hp.weight = [0.2, 0.8]
    elif hp.dataset == "Davis":
        # hp.weight = [1, 2.52
        hp.weight = [0.3, 0.7]

    hp.file_dir = f"result/{hp.dataset}"

    if not os.path.exists(f'{hp.file_dir}'):
        os.makedirs(f'{hp.file_dir}')

    if not os.path.exists("./weight"):
        os.makedirs("./weight")

    fold_num = len(os.listdir(hp.file_dir))
    if os.path.exists(os.path.join(hp.file_dir, "Test")):
        fold_num -= 1
    hp.file_dir = os.path.join(hp.file_dir, str(fold_num))
    os.makedirs(f'{hp.file_dir}')
    hp.filename = hp.current_time

    with open(f'{hp.file_dir}/valid.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        for temp in hp.__dict__.items():
            print(f"{temp[0]:>35}:{temp[1]}")
            writer.writerow(list(temp))

    with open(f'{hp.file_dir}/result.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow("fold,set,thred_optim,loss,auc,aupr,ACC,precision,recall,sp,TP,TN,FP,FN".split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("this code run on GPU!")
    else:
        print("this code run on CPU!")

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(hp.random_seed)

    for i in range(k_fold):
        if i != args.k:
            continue
        # 模型准备
        model = hp.model(return_one=hp.return_one,
                         dropout=hp.dropout,
                         dim=hp.dim,
                         conv=hp.conv,
                         n_layer=hp.n_layer,
                         protein_kernel_size=hp.protein_kernel_size,
                         k=hp.topk)

        if hp.load_weight is not None:
            print("加载预训练权重")
            model.load_state_dict(torch.load(hp.load_weight))

        # 数据准备
        print(f"正在跑第{i + 1}折")

        train_fold_dataloader = DataLoader(
            dataset=DTIDataset(return_drug_info=hp.return_drug_info, dataset=hp.dataset,
                               data_dir=f"kfolddata/{hp.dataset}/{args.e}/{i}/train.csv"),
            batch_size=hp.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        valid_fold_dataloader = DataLoader(
            dataset=DTIDataset(return_drug_info=hp.return_drug_info, dataset=hp.dataset,
                               data_dir=f"kfolddata/{hp.dataset}/{args.e}/{i}/val.csv"),
            batch_size=hp.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        test_dataloader = DataLoader(
            dataset=DTIDataset(return_drug_info=hp.return_drug_info, dataset=hp.dataset,
                               data_dir=f"kfolddata/{hp.dataset}/{args.e}/{i}/test.csv"),
            batch_size=hp.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # 优化器 + L2正则化 + 权重初始化
        weight_p, bias_p = [], []
        # for p in model.parameters():
        #     if p.dim() > 1:
        #         torch.nn.init.xavier_uniform_(p, gain=1)
        #         # nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
            lr=hp.lr)
        # 动态学习率
        scheduler = None
        # 损失函数
        if hp.return_one:
            loss_fn = torch.nn.BCELoss()
        else:
            weight = None
            if hp.weight is not None:
                weight = torch.FloatTensor(hp.weight).to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        # 训练用对象
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataset=train_fold_dataloader,
            valid_dataset=valid_fold_dataloader,
            use_cuda=True if device == "cuda" else False,
            hyperparamter=hp
        )
        _ = trainer.run()

        train_loss, (_, train_result) = trainer.evaluation(train_fold_dataloader, load_model=True,
                                                           filename=f"{hp.filename}.pt")
        print(f"train:{round(train_loss, 4)}, {train_result}")
        valid_loss, (thred_optim, valid_result) = trainer.evaluation(valid_fold_dataloader, load_model=True,
                                                                     filename=f"{hp.filename}.pt")
        print(f"valid:{round(valid_loss, 4)}, {valid_result}")
        test_loss, (_, test_result) = trainer.evaluation(test_dataloader, load_model=True, filename=f"{hp.filename}.pt",
                                                         thred_optim=thred_optim)
        print(f" test:{round(test_loss, 4)}, {test_result}")
        with open(f"{hp.file_dir}/result.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow([i + 1, 'train', "None", round(train_loss, 4)] + train_result)
            writer.writerow([i + 1, 'valid', thred_optim, round(valid_loss, 4)] + valid_result)
            writer.writerow([i + 1, ' test', thred_optim, round(test_loss, 4)] + test_result)

        model_size = round(os.path.getsize(f"weight/{hp.filename}.pt") / (1024 ** 2), 3)
        print(f"模型文件大小：{model_size}Mb.")
        if hp.save_model:
            import shutil
            print(f"模型保存到{hp.file_dir}/{i + 1}.pt")
            shutil.move(f"weight/{hp.filename}.pt", f'{hp.file_dir}/{i + 1}.pt')
        else:
            os.remove(f"weight/{hp.filename}.pt")

        del trainer, model, valid_fold_dataloader, train_fold_dataloader, \
            optimizer, loss_fn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
