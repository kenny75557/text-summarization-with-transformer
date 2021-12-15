import torch
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from tqdm.auto import tqdm


class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=T5ForConditionalGeneration.from_pretrained(MODEL_NAME,return_dict=True)
    def forward(self,input_ids,attention_mask,decoder_attention_mask,labels=None):
        output=self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss,output.logits
    def training_step(self,batch,batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
        
        loss,outputs=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )
        self.log("train_loss",loss,prog_bar=True,logger=True)
        return loss
    def validation_step(self,batch,batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]

        loss,outputs=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )
        self.log("val_loss",loss,prog_bar=True,logger=True)
        return loss
    def test_step(self,batch,batch_idx):
        input_ids=batch["text_input_ids"]
        attention_mask=batch["text_attention_mask"]
        labels=batch["labels"]
        labels_attention_mask=batch["labels_attention_mask"]
        
        loss,outputs=self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )
        self.log("test_loss",loss,prog_bar=True,logger=True)
        return loss
    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=0.0001)