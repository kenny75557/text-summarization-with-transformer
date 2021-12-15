#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import cuda
from data_modules import NewsSummaryDataModule
from models import NewsSummaryModel
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from sklearn.model_selection import train_test_split

DEVICE_COUNT = cuda.device_count()
PROJECT="News-Summary"


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=120)
    # parser.add_argument(
    #     "--mode", type=str, required=True
    # )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--num_epochs", type=int, default=3)
    # parser = Trainer.add_argparse_args(parser)
    # parser = NewsSummaryModel.add_model_specific_args(parser)
    # parser = NewsSummaryDataModule.add_argparse_args(parser)
    # parser.set_defaults(
    #     accelerator="dp",
    #     auto_select_gpus=True,
    #     batch_size=16,
    #     gpus=DEVICE_COUNT,
    #     max_epochs=10,
    #     model_name_or_path="facebook/bart-base",
       
    # )




def init_setup(args):
    seed_everything(args.seed)
    #News-Summary Dataset
    df=pd.read_csv("/user_data/Text-Summarization-with-pytorchLightning/data_modules/news_summary.csv",encoding="latin-1")
    df=df[["text","ctext"]]
    df.columns=["summary","text"]
    df=df.dropna()
    train_df,test_df=train_test_split(df,test_size=0.1)
    tokenizer=T5Tokenizer.from_pretrained(args.model_name_or_path)
    data_module=NewsSummaryDataModule(train_df,test_df,tokenizer,batch_size=args.batch_size)

    model=NewsSummaryModel(args,tokenizer)

    checkpoint_callback=ModelCheckpoint(
        dirpath=f"checkpoints/{PROJECT}/",
        filename="summary-checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger=[TensorBoardLogger("lightning_logs",name="news-summary"),WandbLogger(project=PROJECT)]
   
    trainer=pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.num_epochs,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    return model, data_module, trainer




def main(args):
    model, data_module, trainer = init_setup(args)

    #training
    trainer.fit(model,data_module)

    #testing



if __name__=="__main__":
    parser=build_arg_parse()
    args=parser.parse_args()
    main(args)
