import csv

import numpy as np
import sentencepiece as spm
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset


def read_csv(file_path, keys: list) -> list:
    """
    用于读取csv文件，通过传去的keys将每一行转换为字典后输出
    :param file_path:
    :param keys:
    :return:
    """
    result = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) == 1:
                line = line[0].split(" ")
            if len(keys) != len(line):
                raise ValueError(f"{keys}")
            temp = {}
            for i in range(len(keys)):
                temp[keys[i]] = line[i]
            result.append(temp)
    return result


smiles_sorted = {'X': 0, 'C': 1, '=': 2, '(': 3, ')': 4, 'O': 5, '1': 6, '@': 7, '[': 8, ']': 9, 'N': 10,
                 'H': 11, '2': 12, '3': 13, 'F': 14, 'S': 15, '\\': 16, '4': 17, 'l': 18, 'P': 19, '+': 20,
                 '-': 21, '/': 22, '.': 23, '5': 24, '#': 25, 'B': 26, 'r': 27, 'I': 28, '6': 29, 'a': 30,
                 '8': 31, 'e': 32, '%': 33, '7': 34, 'u': 35, 'n': 36, 'A': 37, 'Z': 38, 'g': 39, '9': 40,
                 'i': 41, 'M': 42, 'K': 43, 'L': 44, 's': 45, '0': 46, 'o': 47, 'G': 48, 'd': 49, 'T': 50,
                 'c': 51, 't': 52, 'b': 53, 'V': 54, 'W': 55, 'R': 56, 'm': 57}

seq_sorted = {'X': 0, 'L': 1, 'A': 2, 'G': 3, 'S': 4, 'V': 5, 'E': 6, 'K': 7, 'T': 8, 'I': 9, 'P': 10,
              'R': 11, 'D': 12, 'F': 13, 'N': 14, 'Q': 15, 'Y': 16, 'M': 17, 'H': 18, 'C': 19, 'W': 20, 'U': 21}


class DTIDataset(Dataset):
    def __init__(self,
                 dataset="DrugBank",
                 return_drug_info=None,
                 drug_atom_max_len=50,
                 drug_max_len=100,
                 target_max_len=1000,
                 data_dir=None):
        if return_drug_info is None:
            return_drug_info = ['morgan', 'AtomFeature', 'CoulombMatrix', 'smilesEncode', "molGraph", "bpeEncode"]
        self.return_drug_info = return_drug_info
        self.drug_atom_max_len = drug_atom_max_len
        self.drug_max_len = drug_max_len
        self.target_max_len = target_max_len
        if dataset == "Davis":
            data = f"data/Davis/Davis.csv"
        elif dataset == "KIBA":
            data = f"data/KIBA/KIBA.csv"
        elif dataset == "DrugBank":
            data = f"data/DrugBank/Post-processing Data/0/0.csv"
        else:
            raise ValueError(f"{dataset} is not not available to get!")

        if data_dir is not None:
            data = data_dir

        self.results = read_csv(data, ["drug_id", "target_id", "SMILES", "target_seq", "label"])

        if "morgan" in self.return_drug_info:
            self.morgan_list = np.load(f"./data/{dataset}/fingerprintFromSmiles.npy", allow_pickle=True).item()

        if "AtomFeature" in self.return_drug_info:
            self.atom_list = np.load(f"./data/{dataset}/moleculeFromSmiles.npy", allow_pickle=True).item()

        if "CoulombMatrix" in self.return_drug_info:
            self.coulomb_list = np.load(f"./data/{dataset}/coulombMatrixFromSmiles.npy", allow_pickle=True).item()

        if "molGraph" in self.return_drug_info:
            self.graph_list = load_graphs(f"./data/{dataset}/graphFromSmiles.npy")

        if "bpeEncode" in self.return_drug_info:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load('./data/bpe1/m.model')

    def __len__(self):
        '''返回数据集中的样本数'''
        return len(self.results)

    def encodeDrug(self, drug_id, SMILES):
        result = []
        for return_info in self.return_drug_info:
            if return_info == "morgan":
                result.append(self.morgan_list[drug_id])
            elif return_info == "AtomFeature":
                temp = np.array(self.atom_list[drug_id])
                temp = np.vsplit(temp, [self.drug_atom_max_len])[0]
                temp = np.concatenate((temp, np.zeros([self.drug_atom_max_len - temp.shape[0], 28])), axis=0)
                result.append(temp)
            elif return_info == "CoulombMatrix":
                coulombMatrix = self.coulomb_list[drug_id]
                temp = np.hsplit(coulombMatrix, [self.drug_atom_max_len])[0]
                temp = np.vsplit(temp, [self.drug_atom_max_len])[0]
                temp = np.pad(temp, (
                    (0, self.drug_atom_max_len - temp.shape[0]), (0, self.drug_atom_max_len - temp.shape[1])))
                result.append(temp)
            elif return_info == "smilesEncode":
                temp = np.zeros(self.drug_max_len, dtype=np.int64())
                for i, ch in enumerate(SMILES[:self.drug_max_len]):
                    temp[i] = smiles_sorted[ch]
                result.append(temp)
            elif return_info == "molGraph":
                index = int(self.graph_list[1][drug_id])
                graph = self.graph_list[0][index]
                result.append(graph)
            elif return_info == "bpeEncode":
                temp = np.array(self.sp.encode_as_ids(SMILES)[:50])
                zeros = np.zeros(50 - len(temp))
                result.append(np.concatenate([temp, zeros]))
            else:
                raise ValueError(f"{return_info} is not available to get!")
        return result

    def encodeTarget(self, target_seq):
        seq = target_seq[:self.target_max_len]
        temp = np.zeros(self.target_max_len, dtype=np.int64())
        for i, ch in enumerate(seq):
            temp[i] = seq_sorted[ch]
        return temp

    def __getitem__(self, index):
        '''获取数据的方法，会和Dataloader连用'''
        if index is None:
            return [int(temp["label"]) for temp in self.results]
        record = self.results[index]
        drug = self.encodeDrug(record["drug_id"], record["SMILES"])
        target = self.encodeTarget(record["target_seq"])
        label = int(record["label"])

        return drug, target, label
