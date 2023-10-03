# -*- coding: utf-8 -*-
from datetime import datetime


class Hyperparameter:
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.model = None

        self.dataset = "DrugBank"
        self.lr = 1e-4
        self.epochs = 200
        self.batch_size = 64
        self.patience = 25
        self.dropout = 0.5
        self.weight_decay = 0.06
        self.conv = 64
        self.dim = 64
        self.n_layer = 2
        self.topk = 1

        if self.n_layer == 2:
            self.protein_kernel_size = [9, 13]

        self.weight = None

        self.lr_decay = 0.5
        self.random_seed = 1234
        self.return_one = False
        self.start_early_stopping_epoch = 0

        self.paint_loss = True
        self.save_model = True
        self.file_dir = None
        self.filename = None

        self.return_drug_info = ['CoulombMatrix', "molGraph", "smilesEncode", 'morgan']
