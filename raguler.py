import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc as Auc
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, \
    precision_score, recall_score, accuracy_score


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def onehot(data, classes=-1):
    matrix = np.eye(classes)
    if isinstance(data, int):
        return matrix[data]

    if classes == -1:
        classes = max(data) + 1
    results = []
    for i in data:
        results.append(matrix[i])

    return np.array(results)


def splitSeq(seq, k):
    seq = seq.strip().strip("\n")
    temp = seq[0:k]
    for i in range(1, len(seq) - (k - 1)):
        temp += " "
        temp += seq[i: i + k]
    return temp


def read_fasta(path):
    results = []
    with open(path, "r") as file:
        sample = {}
        seq = ""
        for line in file.readlines():
            if ">" in line:
                sample["seq"] = seq
                seq = ""
                results.append(sample)
                name = line[1:].split("|")[1]
                sample = {"name": name}
            else:
                seq += line.strip().strip("\n")
    sample["seq"] = seq
    results.append(sample)
    results = results[1:]
    return results


def evaluate(predict_scores, labels, ndigits=0, thred_optim=None):
    labels = np.array(labels)
    auc = aupr = accuracy = precision = recall = specificity = TP = TN = FP = FN = None

    if np.all(labels):
        predict_scores = F.softmax(torch.tensor(predict_scores), dim=1).numpy()
        predict_scores = predict_scores[:, 1]
        predict_labels = [1 if i >= thred_optim else 0 for i in predict_scores]
        return None, None, predict_scores, predict_labels

    if len(predict_scores.shape) == 2:
        predict_scores = F.softmax(torch.tensor(predict_scores), dim=1).numpy()
        # predict_labels = np.argmax(predict_scores, axis=1)
        predict_scores = predict_scores[:, 1]
        tpr, fpr, thresholds = precision_recall_curve(labels, predict_scores)
        aupr = Auc(fpr, tpr)
        auc = roc_auc_score(labels, predict_scores)
        if thred_optim is None:
            f1 = 2 * tpr * fpr / (fpr + tpr + 0.0000000000001)
            thred = thresholds[np.argmax(f1)]
            thred_optim = thred
    else:
        predict_scores = torch.nn.Sigmoid()(predict_scores)
        tpr, fpr, thresholds = precision_recall_curve(labels, predict_scores)
        aupr = Auc(fpr, tpr)
        auc = roc_auc_score(labels, predict_scores)
        # predict_labels = [1 if i else 0 for i in (predict_scores >= 0.5)]

        if thred_optim is None:
            f1 = 2 * tpr * fpr / (fpr + tpr + 0.0000000000001)
            thred = thresholds[np.argmax(f1)]
            thred_optim = thred

    predict_labels = [1 if i else 0 for i in (predict_scores >= thred_optim)]
    [[TN, FP], [FN, TP]] = confusion_matrix(labels, predict_labels)
    specificity = TN / (FP + TN)
    precision = precision_score(labels, predict_labels)
    recall = recall_score(labels, predict_labels)
    accuracy = accuracy_score(labels, predict_labels)

    results = [auc, aupr, accuracy, precision, recall, specificity, int(TP), int(TN), int(FP), int(FN)]
    if ndigits > 0:
        temp = [round(i, ndigits) for i in results[:-4]]
        results = temp + results[-4:]
    return thred_optim, results, None, None


class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = F.one_hot(labels, num_classes=2).to(device=self.DEVICE)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1 - pt)
        return torch.mean(poly1)
