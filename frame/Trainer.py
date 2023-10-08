import numpy as np
import torch
from matplotlib import pyplot as plt

from frame.EarlyStopping import EarlyStopping
from hyperparameter import Hyperparameter
from raguler import evaluate


class Trainer(object):
    def __init__(self, model, loss_fn, optimizer, train_dataset, valid_dataset, use_cuda, hyperparamter: Hyperparameter,
                 scheduler=None):
        self.hp = hyperparamter
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.use_cuda = use_cuda
        self.paint_loss = hyperparamter.paint_loss
        self.iteration = 0
        self.earlyStopping = EarlyStopping(self,
                                           patience=hyperparamter.patience,
                                           start_epoch=hyperparamter.start_early_stopping_epoch,
                                           filename=hyperparamter.filename)

        if self.use_cuda:
            self.model = self.model.cuda()
        self.buff = {
            "train_loss": [],
            "val_loss": []
        }

    def run(self):
        epochs = self.hp.epochs
        for i in range(epochs):
            lr = self.optimizer.param_groups[0]['lr']
            train_loss, train_result = self.train()
            val_loss, (_, val_result, _, _) = self.evaluation(self.valid_dataset)
            val_result = [val_loss] + val_result

            self.buff["train_loss"].append(train_loss)
            self.buff["val_loss"].append(val_loss)
            epoch_len = len(str(epochs))
            print_msg = (f'[{i + 1:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         f'lr: {lr:.5f} ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {val_loss:.5f} ' +
                         f'valid_AUC: {val_result[1]:.5f} ' +
                         f'valid_AUPR: {val_result[2]:.5f} ' +
                         f'valid_Acc: {val_result[3]:.5f} ' +
                         f'valid_Precision: {val_result[4]:.5f} ' +
                         f'valid_Reacll: {val_result[5]:.5f} ' +
                         f'valid_Sp: {val_result[6]:.5f} ')
            print(print_msg)
            self.earlyStopping(val_loss, i, val_result)
            if self.earlyStopping.early_stop:
                break
        # 画图
        if self.paint_loss:
            self.func_paint_loss()
        return self.earlyStopping.best_record

    def train(self):
        self.model.train()
        epoch_loss = 0
        outputs = None
        labels = None

        for i, (drug, target, label) in enumerate(self.train_dataset, start=1):
            if self.use_cuda:
                temp = []
                for x in range(len(drug)):
                    temp.append(drug[x].to(torch.device("cuda")))
                drug = temp
                target = target.cuda()
                label = label.cuda()

            self.optimizer.zero_grad()
            output = self.model(drug, target)
            if self.hp.return_one:
                output = torch.squeeze(torch.nn.Sigmoid()(output))
                loss = self.loss_fn(output, label.float())
            else:
                loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.step()

            # Loss-value superposition and computation of confusion matrices
            epoch_loss += loss.item()
            if outputs is None:
                outputs = output.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                outputs = np.concatenate([outputs, output.cpu().detach().numpy()], axis=0)
                labels = np.concatenate([labels, label.cpu().detach().numpy()], axis=0)

        self.iteration += i
        epoch_loss /= i

        if self.scheduler:
            self.scheduler.step()

        return epoch_loss, evaluate(outputs, labels, ndigits=4)

    def evaluation(self, data, load_model=False, filename="temp.pt", thred_optim=None):
        self.model.eval()
        eval_loss = 0
        outputs = None
        labels = None

        if load_model is True:
            self.model.load_state_dict(torch.load(f"weight/{filename}"))

        for i, (drug, target, label) in enumerate(data, start=1):
            if self.use_cuda:
                temp = []
                for x in range(len(drug)):
                    temp.append(drug[x].to(torch.device("cuda")))
                drug = temp
                target = target.cuda()
                label = label.cuda()
            with torch.no_grad():
                out = self.model(drug, target)
                if self.hp.return_one:
                    out = torch.squeeze(torch.nn.Sigmoid()(out))
                    loss = self.loss_fn(out, label.float())
                else:
                    loss = self.loss_fn(out, label)
                eval_loss += loss.item()

            if outputs is None:
                outputs = out.cpu().detach().numpy()
                labels = label.cpu().detach().numpy()
            else:
                outputs = np.concatenate([outputs, out.cpu().detach().numpy()], axis=0)
                labels = np.concatenate([labels, label.cpu().detach().numpy()], axis=0)
        eval_loss /= i

        return eval_loss, evaluate(outputs, labels, ndigits=4, thred_optim=thred_optim)

    def func_paint_loss(self):
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        train_loss = self.buff["train_loss"]
        val_loss = self.buff["val_loss"]
        x = range(1, len(train_loss) + 1)
        plt.figure(dpi=80)

        plt.plot(x, train_loss, label="train", color="red")
        plt.plot(x, val_loss, label="val", color="blue")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.savefig(f"{self.hp.file_dir}/loss.png")

        train_loss = self.buff["train_loss"][5:]
        val_loss = self.buff["val_loss"][5:]
        x = range(1, len(train_loss) + 1)
        plt.figure(dpi=80)

        plt.plot(x, train_loss, label="train", color="red")
        plt.plot(x, val_loss, label="val", color="blue")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(f"{self.hp.file_dir}/loss_less_5.png")
