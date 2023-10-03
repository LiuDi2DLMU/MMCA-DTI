# -*- coding: utf-8 -*-
import os.path

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


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
        raise Exception(f"合并时有问题 {need_negetive} // {num_negetive}, {need_positive} // {num_positive}")

    return a.drop_duplicates()


def dropdata(data, keys, position, radio=0.5):
    result = []
    for i, key in enumerate(keys):
        temp = data[data[position] == key]
        if len(temp) == 0:
            raise ValueError(f"{key} 不存在")
        if len(temp) >= 2:
            temp = temp.sample(frac=radio)
        result.append(temp)
    result = pd.concat(result)
    return result


def split_k_fold(kfold, dataset="DrugBank", e=2):
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

    if e != 2 and e != 3:
        raise ValueError(f"e {e} is wrong!")

    data = pd.read_csv(file, delimiter=delimiter, header=None).drop_duplicates()
    prop = len(data.loc[data[4] == 1]) / len(data.loc[data[4] == 0])
    kf = KFold(n_splits=kfold, shuffle=True, random_state=1)
    print(prop, "\n")
    data_x = data[e - 2].drop_duplicates()
    for i, (train_index, test_index) in enumerate(kf.split(data_x)):
        test_data = data_x.iloc[test_index]
        other_data = data_x.iloc[train_index]

        test_data = data.loc[data[e - 2].isin(list(test_data))]
        other_data = data.loc[data[e - 2].isin(list(other_data))]
        print(i, len(test_data.loc[test_data[4] == 1]) / len(test_data.loc[test_data[4] == 0]),
              len(other_data.loc[other_data[4] == 1]) / len(other_data.loc[other_data[4] == 0]))
        train_size = int(len(other_data) * 0.8)

        train_data, val_data, _, _ = train_test_split(other_data, other_data[4], test_size=0.2, random_state=1,
                                                      stratify=other_data[4])

        if not os.path.exists(f"kfolddata/{dataset}/{e}/{i}/"):
            os.makedirs(f"kfolddata/{dataset}/{e}/{i}/")
        train_data.to_csv(f"kfolddata/{dataset}/{e}/{i}/train.csv", header=False, index=False)
        val_data.to_csv(f"kfolddata/{dataset}/{e}/{i}/val.csv", header=False, index=False)
        test_data.to_csv(f"kfolddata/{dataset}/{e}/{i}/test.csv", header=False, index=False)
    file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default="DrugBank", help="The dataset used for training")
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument("--e", type=int, default=2, help="Which dataset settings to load")
    args = parser.parse_args()

    split_k_fold(args.k, dataset=args.dataset, e=args.e)
