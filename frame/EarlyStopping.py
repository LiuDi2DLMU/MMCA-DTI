import os

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, trainer, patience=15, delta=0, start_epoch=0, filename='temp.pt'):
        if filename is None:
            filename = "temp.pt"
        if '.pt' not in filename:
            filename = filename + '.pt'

        self.filename = filename
        self.trainer = trainer
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_record = None
        self.start_epoch = start_epoch

    def __call__(self, val_loss, epoch, result):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
            self.best_record = result
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_record = result
            self.save_checkpoint()
            self.counter = 0

        if epoch < self.start_epoch:
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self):
        """
        Saves model when validation loss decrease.
        """
        path = os.path.join("weight", self.filename)
        torch.save(self.trainer.model.state_dict(), path)
        self.val_loss_min = self.best_score
