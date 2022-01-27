#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 01/24/22 10:38 PM
# @Author  : Fabrice Harel-Canada
# @File    : minist.py

import os
from typing import Optional
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam

import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from pytorch_lightning import Trainer, LightningDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def unstable_softmax(logits):
    """Computes softmax in a numerically unstable way."""
    exp = torch.exp(logits)
    sm = exp / torch.sum(exp)
    return sm

def unstable_cross_entropy(probs, labels):
    """Computes cross entropy in a numerically unstable way."""
    return -torch.sum(torch.log(probs) * labels)
    
def stable_cross_entropy(logits, labels, reduction='mean'):
    batchloss = -torch.sum(labels.squeeze() * torch.log(logits), dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')

class StableModel(LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, self.num_classes)
        
        # metrics
        self.accuracy = torchmetrics.Accuracy()
        self.accuracy.mode = "multi-label"

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        
        # log step metric
        loss = stable_cross_entropy(probs, y)
        
        self.accuracy(probs, y)
        self.log('train_acc_step', self.accuracy)
        
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy)
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        
        # log step metric
        loss = stable_cross_entropy(probs, y)
        self.log("val_loss", loss)
        
        self.accuracy(probs, y)
        self.log('val_acc_step', self.accuracy)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        
        # log step metric
        loss = stable_cross_entropy(probs, y)
        self.log("test_loss", loss)
        
        self.accuracy(probs, y)
        self.log('test_acc', self.accuracy)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def predict(self, x):
        out = self(x)
        if torch.isnan(out).any():
            raise ValueError("Model prediction contains NaN.")
        return torch.argmax(out)
    
class UnstableSMModel(StableModel):
    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = unstable_softmax(x)
        return x
    
class UnstableCEModel(StableModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = unstable_cross_entropy(probs, y)
        return loss
    
class UnstableSMCEModel(StableModel):
    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = unstable_softmax(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = unstable_cross_entropy(probs, y)
        return loss


class MNISTDataModule(LightningDataModule):
    def __init__(self, 
                 data_dir: Optional[str] = './data/', 
                 batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.target_transforms = transforms.Compose([
            lambda x:torch.LongTensor([x]),
            lambda x:F.one_hot(x, 10).squeeze()
        ])
        self.setup()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, 
                               train=True, 
                               transform=self.transforms,
                               target_transform=self.target_transforms)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, 
                               train=False, 
                               transform=self.transforms,
                               target_transform=self.target_transforms)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def load_models():

    sm = StableModel()
    usmm = UnstableSMModel()
    ucem = UnstableCEModel()
    usmcem = UnstableSMCEModel()

    models = [sm, usmm, ucem, usmcem]
    
    return_models = []
    for model in models:
        
        model_name = model.__class__.__name__
        model_save_path = os.path.join('./pretrained/', model_name + '.pth')
        
        if os.path.exists(model_save_path):
            print("Loading pretrained %s model!" % (model_name))
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
        else:
            raise FileNotFoundError(
                "A pretrained %s model was not found \
                - run `python mnist.py` to generate!" % (model_name)
            )
        return_models.append(model)
    return return_models


if __name__ == '__main__':

    sm = StableModel()
    usmm = UnstableSMModel()
    ucem = UnstableCEModel()
    usmcem = UnstableSMCEModel()

    models = [sm, usmm, ucem, usmcem]
    mnist_dm = MNISTDataModule()

    escb = EarlyStopping(monitor="train_acc_epoch", mode="max", patience=3)

    results = []
    for model in models:
        
        model_name = model.__class__.__name__
        model_save_path = os.path.join('./pretrained/', model_name + '.pth')
        
        if os.path.exists(model_save_path):
            print("Loading pretrained %s model!" % (model_name))
            model.load_state_dict(torch.load(model_save_path))
            model.eval()
        else:
            # train
            print("Training a new %s model!" % (model_name))
            trainer = Trainer(max_epochs=10, callbacks=[escb])
            trainer.fit(model, datamodule=mnist_dm)

            # save model
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            
            model.eval()
        
        
        trainer = Trainer(max_epochs=10, callbacks=[escb])  
        out = trainer.test(model, datamodule=mnist_dm)
        out[0]['model'] = model
        out[0]['model_name'] = model_name
        results.extend(out)

    import requests
    from PIL import Image
    import cv2

    # image of a 2
    url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    img.show()

    img_array = np.asarray(img)
    resized = cv2.resize(img_array, (28, 28))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # (28, 28)
    image = cv2.bitwise_not(gray_scale)

    image = image / 255
    image = image.reshape(1, 28, 28)
    image = torch.from_numpy(image).float().unsqueeze(0)

    for result in results:
        print(result['model'].predict(image))
