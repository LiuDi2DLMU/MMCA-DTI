# -*- coding: utf-8 -*-
import csv
import sys
from io import StringIO

import numpy as np
import torch
from dgl import DGLGraph
from dgl.data.utils import save_graphs
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem

rdBase.LogToPythonStderr()


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


def one_of_k_encoding_unk(x, allowable_set):
    'Compare x with allowable_set one by one, same is True, different is False, all different is considered to be the last same.'
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(np.array(list(map(lambda s: x == s, allowable_set))) + 0)


def get_atom_features(atom, xyz):
    possible_atom = ["C", "O", "N", "F", "S", "Cl", "P", "Br", "I", "Si", 'H']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])  # Implicit valence
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [0, -1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                           [Chem.rdchem.HybridizationType.S,
                                            Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3])
    atom_features += list(xyz)
    return np.array(atom_features)


def get_coulombmatrix(xyzs, atoms_atomic_numbers, largest_mol_size=None):
    """
    This function generates a coulomb matrix for the given molecule
    if largest_mol size is provided matrix will have dimension lm x lm.
    Padding is provided for the bottom and right _|
    """
    numberAtoms = len(atoms_atomic_numbers)
    if largest_mol_size == None or largest_mol_size == 0: largest_mol_size = numberAtoms

    if numberAtoms > largest_mol_size:
        numberAtoms = largest_mol_size

    cij = np.zeros((largest_mol_size, largest_mol_size))

    xyzmatrix = [[xyz[0], xyz[1], xyz[2]] for xyz in xyzs]
    chargearray = [atomic_number for atomic_number in atoms_atomic_numbers]

    for i in range(numberAtoms):
        for j in range(numberAtoms):
            if i == j:
                cij[i][j] = 0.5 * chargearray[i] ** 2.4  # Diagonal term described by Potential energy of isolated atom
            else:
                dist = np.linalg.norm(np.array(xyzmatrix[i]) - np.array(xyzmatrix[j]))
                cij[i][j] = chargearray[i] * chargearray[j] / dist  # Pair-wise repulsion
    return cij


def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)


def read_csv(file_path, keys: list) -> list:
    """
    Used to read CSV files, convert each line into a dictionary through the passed keys, and output
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
              'R': 11, 'D': 1, 'F': 13, 'N': 14, 'Q': 15, 'Y': 16, 'M': 17, 'H': 18, 'C': 19, 'W': 20, 'U': 21}


def encodeTarget(target_seq, MAX_LEN=1000):
    seq = target_seq[:MAX_LEN]
    temp = np.zeros(MAX_LEN, dtype=np.int64())
    for i, ch in enumerate(seq):
        temp[i] = seq_sorted[ch]
    return temp


def encodeDrug(SMILES):
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        raise Exception("mol is None")
    mol = Chem.AddHs(mol)
    state = AllChem.EmbedMolecule(mol, useRandomCoords=True)
    info = sio.getvalue()
    if mol is None or state == -1:
        raise Exception(f"state is {state}, EmbedMolecule fail!")
    if info != "":
        print(info)
    mol = Chem.RemoveAllHs(mol)
    # Coordinates and atomic information
    conf = mol.GetConformer()
    geom = list(conf.GetPositions())
    if len(geom) == 0:
        raise Exception(f"len of geom is 0!")
    G = DGLGraph()
    G.add_nodes(mol.GetNumAtoms())
    atom_features = []
    atom_AtomicNum = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        anum = atom_i.GetAtomicNum()
        atom_features.append(get_atom_features(atom_i, geom[i]))
        atom_AtomicNum.append(anum)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i, j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    G.ndata['x'] = torch.from_numpy(np.array(atom_features))  # Adding atomic/node features to dgl
    G.edata['w'] = torch.from_numpy(np.array(edge_features))  # Add key/edge features to dgl

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, useFeatures=True).ToBitString()
    fp = np.array(list(map(int, list(fp))))
    atom_features = np.array(atom_features)
    coulombMatrix = get_coulombmatrix(geom, atom_AtomicNum)

    MAX_ATOM = 50
    temp = np.array(atom_features)
    temp = np.vsplit(temp, [MAX_ATOM])[0]
    temp = np.concatenate((temp, np.zeros([MAX_ATOM - temp.shape[0], 28])), axis=0)
    atom_features = temp

    temp = np.hsplit(coulombMatrix, [MAX_ATOM])[0]
    temp = np.vsplit(temp, [MAX_ATOM])[0]
    temp = np.pad(temp, (
        (0, MAX_ATOM - temp.shape[0]), (0, MAX_ATOM - temp.shape[1])))
    coulombMatrix = temp

    return fp, atom_features, coulombMatrix, G


def get_data(dataset):
    if dataset == "Davis":
        data_dir = f"data/Davis/Davis.csv"
    elif dataset == "KIBA":
        data_dir = f"data/KIBA/KIBA.csv"
    elif dataset == "DrugBank":
        data_dir = f"data/DrugBank/Post-processing Data/0/0.csv"
    else:
        raise ValueError(f"dataset {dataset} is not available to get!")
    return read_csv(data_dir, ["drug_id", "target_id", "SMILES", "target_seq", "label"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default="DrugBank", help="The dataset used for training")
    args = parser.parse_args()
    dataset = args.dataset

    data = get_data(dataset)
    temp = {}
    for x in data:
        drug_id = x["drug_id"]
        if drug_id not in temp.keys():
            temp[drug_id] = x["SMILES"]
    data = temp
    fingerprint = {}
    molecule = {}
    coulombMatrix = {}
    drug_dict = {}
    m = 0
    moleculeGraph = [{}, []]
    lens = len(data)
    for i, drug_id in enumerate(data):
        print(f"{i:0>5}/{lens}")
        if drug_id not in drug_dict.keys():
            drug_dict[drug_id] = data[drug_id]
            fp, atom_fea, coulomb, graph = encodeDrug(data[drug_id])
            fingerprint[drug_id] = fp
            molecule[drug_id] = atom_fea
            coulombMatrix[drug_id] = coulomb
            moleculeGraph[0][drug_id] = torch.Tensor([m])
            m += 1
            moleculeGraph[1].append(graph)

    save_graphs(f"data/{dataset}/graphFromSmilesx.npy", moleculeGraph[1], moleculeGraph[0])
    np.save(f"data/{dataset}/moleculeFromSmiles.npy", molecule)
    np.save(f'data/{dataset}/coulombMatrixFromSmiles.npy', coulombMatrix)
    np.save(f"data/{dataset}/fingerprintFromSmiles.npy", fingerprint)
