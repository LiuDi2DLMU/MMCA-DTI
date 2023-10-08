# -*- coding: utf-8 -*-
"""
此文件用于去掉没有数据的部分，以及根据positive数据集搭建negative数据集
"""
import csv
import random

import numpy as np
from Bio import SeqIO
from rdkit import rdBase
from sklearn.utils import shuffle

rdBase.LogToPythonStderr()


def write_csv(content: list, file_path: str) -> None:
    """
    This is a function to write to a csv file
    :param content:  The contents of a line added to the end of the file
    :param file_path:  Path to the file
    :return:
    """
    with open(file_path, "a") as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(content)


def read_csv(file_path, keys: list) -> list:
    """
    Used to read the csv file, through the passed keys will be converted to a dictionary for each line and then output
    :param file_path:
    :param keys:
    :return:
    """
    result = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for line in reader:
            if len(keys) != len(line):
                raise ValueError(f"{keys}")
            temp = {}
            for i in range(len(keys)):
                temp[keys[i]] = line[i]
            result.append(temp)
    return result


def create_negative_samples(target_ids, drug_ids, positive_pairs, random_seed=25):
    """
    Creating an equal number of negative samples from known positive samples
    :param target_ids:
    :param drug_ids:
    :param positive_pairs:
    :param random_seed: Randomized seeds for controlling repetitive generation
    :return:
    """
    random.seed(random_seed)
    negative_pairs = []
    len_samples = len(positive_pairs)
    len_targets = len(target_ids)
    len_drugs = len(drug_ids)
    for i in range(len_samples):
        condition = False
        while not condition:
            random_targets_index = random.randrange(len_targets)
            random_drugs_index = random.randrange(len_drugs)
            temp_pair = {"drug_id": drug_ids[random_drugs_index], "target_uniport_id": target_ids[random_targets_index]}
            if temp_pair not in positive_pairs:
                negative_pairs.append(temp_pair)
                condition = True
    return negative_pairs


random_seed = 25
dataset = "DrugBank"
positive_pairs = read_csv(f"../data/{dataset}/Origin Data/Positive.csv", ["drug_id", "target_id", "target_uniport_id"])
drugs_smiles = read_csv(f"../data/{dataset}/Origin Data/smiles.csv", ["drug_id", "smiles"])

# Drug_id with smiles Remove nulls and inorganic and very small molecule compounds
# Reference HyperAttentionDTI
smiles = {}
count = 0
allow_rdkit_drug_id = list(np.load(f"../data/{dataset}/Post-processing Data/moleculeFromSmiles.npy", allow_pickle=True).item().keys())

for sample in drugs_smiles:
    # Excluding those without SMILES
    if sample["smiles"] != "":
        # Exclusion of inorganic and very small molecule compounds
        if "C" not in sample["smiles"].replace("Ca", "") or "C" not in sample["smiles"].replace("Cl", ""):
            # print(sample["drug_id"], sample["smiles"])
            count += 1
            continue
        # Excluding structures that cannot be extracted using rdkit
        elif sample["drug_id"] not in allow_rdkit_drug_id:
            count += 1
            continue
        if sample["drug_id"] not in smiles.keys():
            smiles[sample["drug_id"]] = sample["smiles"]
all_drug_ids = list(smiles.keys())

# IDs and sequences of all targets in the dataset
target_seqs = {}
for seq in SeqIO.parse(f"../data/{dataset}/Origin Data/target.fasta", "fasta"):
    if seq.id.split("|")[1] not in target_seqs.keys():
        target_seqs[seq.id.split("|")[1]] = seq.seq
all_target_uniport_ids = list(target_seqs.keys())

positive_drug_ids = []
positive_target_uniport_ids = []
temp_positive_pairs = []
for positive_pair in positive_pairs:
    if positive_pair["drug_id"] in all_drug_ids:
        positive_drug_ids.append(positive_pair["drug_id"])
    if positive_pair["target_uniport_id"] in all_target_uniport_ids:
        positive_target_uniport_ids.append(positive_pair["target_uniport_id"])
    if positive_pair["drug_id"] in all_drug_ids and positive_pair["target_uniport_id"] in all_target_uniport_ids:
        temp_positive_pairs.append(positive_pair)

positive_drug_ids = list(set(positive_drug_ids))
positive_target_uniport_ids = list(set(positive_target_uniport_ids))
positive_pairs = temp_positive_pairs
del temp_positive_pairs

'''
After screening, there are 11000 drug data 4877 target data in the database
Among them, there are 7238 drug data with interaction 4877 target data Total 25670 interactions
'''

# Utilization of all targets Actual use of data from 4877 targets
negative_pairs = create_negative_samples(all_target_uniport_ids, all_drug_ids, positive_pairs, random_seed=random_seed)

x, y = shuffle(positive_pairs + negative_pairs,
               [1] * len(positive_pairs) + [0] * len(negative_pairs),
               random_state=random_seed)
for data, label in zip(x, y):
    drug_id = data["drug_id"]
    target_id = data["target_uniport_id"]
    write_csv([drug_id, target_id, smiles[drug_id], str(target_seqs[target_id]), label],
              f"../data/{dataset}/Post-processing Data/0/0.csv")

print("dataset build finished!")
