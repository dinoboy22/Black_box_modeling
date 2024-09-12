
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import torchmetrics 
from torchmetrics import Metric


class BaselineModel(pl.LightningModule):
    def __init__(self, num_input=11, num_output=1, layers=[32,32,8], dropout=0.1, learning_rate=0.001):
        super().__init__()

        layers.insert(0, num_input)

        self.layers = []
        self.acts = []
        self.bns = []
        self.dos = []

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.bns.append(nn.BatchNorm1d(num_features=layers[i+1]))
            self.acts.append(nn.ReLU())
            self.dos.append(nn.Dropout(dropout))
            # self.acts.append(nn.Softplus())
            self.add_module(f"layer{i}", self.layers[-1]) 
            self.add_module(f"act{i}", self.acts[-1])
            self.add_module(f"bn{i}", self.bns[-1])
            self.add_module(f"do{i}", self.dos[-1])

        self.dropout = nn.Dropout(dropout) 
        self.output = nn.Linear(layers[-1], num_output)

        self.loss_fn = nn.MSELoss()
        self.training_step_outputs = []
        self.learning_rate = learning_rate

        # self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        # self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer, act, bn, do in zip(self.layers, self.acts, self.bns, self.dos):
            x = layer(x)
            x = act(x)
            # x = bn(x)
            x = do(x) 
            # x = do(bn(act(layer(x))))

        # x = self.dropout(x)
        return self.output(x).squeeze()

    def accuracy(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)        
        y_abs = torch.abs(y_true)
        diff = torch.min(diff, y_abs)
        acc = (y_abs - diff) / y_abs
        return torch.mean(acc)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        accuracy = self.accuracy(y_pred, y)
        self.training_step_outputs.append({
            'loss': loss, 
            'accuracy': accuracy,
            # 'y_pred': y_pred, 
            # 'y': y
        })
        accuracy = self.accuracy(y_pred, y)
        # f1_score = self.f1_score(y_pred, y)
        self.log_dict(
            {
                'train_loss': loss, 
                # 'train_f1_score': f1_score
            },
            on_step = True,
            on_epoch = False,
            prog_bar = False,
        )
        # return {'loss': loss, 'y_pred': y_pred, 'y': y}
        return loss
    #
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in self.training_step_outputs]).mean()
        train_rmse = torch.sqrt(avg_loss)
        # y_pred = torch.cat([x['y_pred'] for x in self.training_step_outputs])
        # y = torch.cat([x['y'] for x in self.training_step_outputs])
        # accuracy = self.accuracy(y_pred, y)
        # f1_score = self.f1_score(y_pred, y)
        self.log_dict(
            {
                'train_loss': avg_loss, 
                'train_rmse': train_rmse,
                'train_accuracy': avg_accuracy,
                # 'train_f1_score': f1_score
            },
            prog_bar = True,
        )
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        val_rmse = torch.sqrt(loss)
        accuracy = self.accuracy(y_pred, y)
        # f1_score = self.f1_score(y_pred, y)
        self.log_dict(
            {
                'val_loss': loss, 
                'val_rmse': val_rmse,
                'val_accuracy': accuracy, 
                # 'val_f1_score': f1_score
            },
            prog_bar = True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        test_rmse = torch.sqrt(loss)
        # accuracy = self.accuracy(y_pred, y)
        # f1_score = self.f1_score(y_pred, y)
        self.log_dict({
            'test_loss': loss, 
            'test_rmse': test_rmse
            # 'test_accuracy': accuracy, 
            # 'test_f1_score': f1_score
        })
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y_pred = self(batch) # self.forward(batch)
        # y_pred = torch.argmax(y_pred, dim=1)
        return y_pred

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optim.SGD(
        #     self.parameters(), 
        #     lr=self.learning_rate, 
        #     momentum=0.9, 
        #     weight_decay=0.001
        # )


