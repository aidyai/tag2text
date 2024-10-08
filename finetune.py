# --------------------------------------------------------
# The Tag2Text Model
# Copyright (c) 2023 
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinyu Huang
# --------------------------------------------------------

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from ram.models import tag2text
# import utils
# from utils import cosine_lr_schedule
from ram.data import create_dataset, create_loader

import clip



class Tag2TextModel(pl.LightningModule):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Ensure that tag2text is correctly callable
        self.model = tag2text(
            pretrained=checkpoint,
            image_size=config['image_size'],
            vit=config['vit'],
            vit_grad_ckpt=config['vit_grad_ckpt'],
            vit_ckpt_layer=config['vit_ckpt_layer'],
            tag_list='ram/data/ram_tag_list.txt'
        )
        self.model.label_embed.requires_grad = False
        self.config = config

    def forward(self, image, caption, parse_tag):
        return self.model(image, caption, parse_tag)

    def training_step(self, batch, batch_idx):
        image, _, caption, _, parse_tag = batch
        image = image.to(self.device)
        loss_t2t, loss_tag = self(image, caption, parse_tag)
        loss = loss_t2t + loss_tag / (loss_tag / loss_t2t).detach()
        
        self.log('train_loss_t2t', loss_t2t, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_tag', loss_tag, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()), 
                                      lr=self.config['init_lr'], weight_decay=self.config['weight_decay'])
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cosine_lr_schedule(optimizer, epoch, self.config['max_epoch'], self.config['init_lr'], self.config['min_lr'])),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]


def main():
    # Specified values
    config_path = '/content/tag2text/config/finetune_tag2text.yaml'
    output_dir = 'output/Pretrain'
    checkpoint = ''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42

    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(seed)

    # Create dataset and dataloader
    datasets = [create_dataset(config, min_scale=0.2)]
    print('Number of training samples:', len(datasets[0]))
    
    data_loader = create_loader(
        datasets, 
        samplers=[None], 
        batch_size=[config['batch_size']], 
        num_workers=[4], 
        is_trains=[True], 
        collate_fns=[None]
    )[0]

    # Create model
    model = Tag2TextModel(config, checkpoint=checkpoint)

    # Logger and callbacks
    wandb_logger = WandbLogger(project='Tag2Text')
    checkpoint_callback = ModelCheckpoint(monitor='train_loss_t2t', mode='min', save_top_k=config['checkpoint']['save_top_k'], save_last=config['checkpoint']['save_last'])
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config['max_epoch'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=1 if torch.cuda.is_available() else 0
    )

    # Training
    trainer.fit(model, data_loader)

if __name__ == '__main__':
    main()
