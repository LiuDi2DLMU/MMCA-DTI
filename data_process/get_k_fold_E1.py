# -*- coding: utf-8 -*-
import os.path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def concatdata(a, b, max_num, prop=1.0):
    num_positive = len(b.loc[b[4] == 1])
    num_negetive = len(b) - num_positive

    need_negetive = int(max_num // (prop + 1) - len(a.loc[a[4] == 0]))
    need_positive = int(max_num - max_num // (prop + 1) - len(a.loc[a[4] == 1]))

    if need_negetive <= num_negetive and need_positive <= num_positive:
        x = b.loc[b[4] == 1].sample(n=need_positive, random_state=1)
        y = b.loc[b[4] == 0].sample(n=need_negetive, random_state=1)
        a = pd.concat([a, x, y])
    else:
        raise Exception(f"Problems with the merger {need_negetive} // {num_negetive}, {need_positive} // {num_positive}")

    return a.drop_duplicates()


def dropdata(data, keys, position, radio=0.5):
    result = []
    for i, key in enumerate(keys):
        temp = data[data[position] == key]
        if len(temp) == 0:
            raise ValueError(f"{key} non-existent")
        if len(temp) >= 2:
            temp = temp.sample(frac=radio)
        result.append(temp)
    result = pd.concat(result)
    return result


def split_k_fold(kfold, dataset="DrugBank"):
    if dataset == "DrugBank":
        file = open("data/DrugBank/DrugBank.csv", "r")
        delimiter = ","
    elif dataset == "Davis":
        file = open("data/Davis/Davis.csv", "r")
        delimiter = " "
    elif dataset == "KIBA":
        file = open("data/KIBA/KIBA.csv", "r")
        delimiter = " "
    else:
        raise ValueError(f"dataset {dataset} is wrong!")

    data = pd.read_csv(file, delimiter=delimiter, header=None).drop_duplicates()
    prop = len(data.loc[data[4] == 1]) / len(data.loc[data[4] == 0])
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1)

    for i, (train_index, test_index) in enumerate(kf.split(data, data[4])):
        test_data = data.iloc[test_index]
        otherdata = data.iloc[train_index]

        test_drug = test_data[0].drop_duplicates()
        test_protein = test_data[1].drop_duplicates()

        other_drug = otherdata[0].drop_duplicates()
        other_protein = otherdata[1].drop_duplicates()

        train_size = int(len(otherdata) * 0.8)

        # 在otherdata中不包含在测试集中出现的药物与靶点
        split_drug = []
        split_protein = []
        for id in test_drug:
            if len(otherdata[otherdata[0] == id]) == 0:
                split_drug.append(id)
        for id in test_protein:
            if len(otherdata[otherdata[1] == id]) == 0:
                split_protein.append(id)
        # 将测试集中的数据移除部分
        test_data2 = test_data.loc[~test_data[0].isin(split_drug), :].loc[~test_data[1].isin(split_protein), :]
        loss = len(test_data) - len(test_data2)
        test_data = test_data2
        del test_data2
        # 保存测试集
        if not os.path.exists(f"kfolddata/{dataset}/1/{i}/"):
            os.makedirs(f"kfolddata/{dataset}/1/{i}/")
        test_data.to_csv(f"kfolddata/{dataset}/1/{i}/test.csv", header=False, index=False)
        # 在测试集中出现的药物与靶点
        test_drug = test_data[0].drop_duplicates()
        test_protein = test_data[1].drop_duplicates()

        # 合成训练集
        # 每条记录 或是药物在测试集中 或者是靶点在测试集中
        data_a = pd.concat([otherdata.loc[otherdata[0].isin(test_drug)],
                            otherdata.loc[otherdata[1].isin(test_protein)]]).drop_duplicates()
        if len(data_a) > train_size:
            # 抛弃一部分数据 保持radio比例的数据在data_a中
            x = dropdata(data_a, test_drug, 0)
            y = dropdata(data_a, test_protein, 1)
            data_a = pd.concat([x, y]).drop_duplicates()
        if len(data_a) > train_size:
            raise Exception(f"After a Drop, the data still exceeds the length limit {len(data_a)} // {train_size}")
        data_b = pd.concat([otherdata, data_a, data_a]).drop_duplicates(keep=False)
        train_data = concatdata(data_a, data_b, max_num=train_size, prop=prop)
        val_data = pd.concat([otherdata, train_data, train_data]).drop_duplicates(keep=False)
        train_data.to_csv(f"kfolddata/{dataset}/1/{i}/train.csv", header=False, index=False)
        val_data.to_csv(f"kfolddata/{dataset}/1/{i}/val.csv", header=False, index=False)

        print(f"Total {len(train_data)} // {len(val_data)} // {len(test_data)} entries, where {loss} entries were removed from the test set")

    file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default="DrugBank", help="The dataset used for training")
    parser.add_argument('--k', type=int, default=5)

    args = parser.parse_args()

    split_k_fold(args.k, dataset=args.dataset)
