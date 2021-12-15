import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from tqdm.auto import tqdm



class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        data:pd.DataFrame,
        tokenizer:T5Tokenizer,
        text_max_token_len:int =512,
        summary_max_token_len:int =128
    ):
        self.tokenizer=tokenizer
        self.data=data
        self.text_max_token_len=text_max_token_len
        self.summary_max_token_len=summary_max_token_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index:int):
        data_row=self.data.iloc[index]
        text=data_row["text"]
        
        text_encoding=tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        summary_encoding=tokenizer(
            data_row["summary"],
            max_length=self.text_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        #padding's id goes "0"  ,
        labels=summary_encoding["input_ids"]
        #replace padding token id's of the labels by -100
        labels[labels==0]=-100
        return dict(
            text=text,
            summary=data_row["summary"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )
class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        tokenizer:T5Tokenizer,
        batch_size:int=8,
        text_max_token_len:int=512,
        summary_max_token_len:int=128
    ):
        super().__init__()
        self.train_df=train_df
        self.test_df=test_df
        
        self.batch_size=batch_size
        self.tokenizer=tokenizer
        self.text_max_token_len=text_max_token_len
        self.summary_max_token_len=summary_max_token_len
    
    def setup(self,stage=None):
        self.train_dataset=NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        self.test_dataset=NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
    def val_dataloader(self):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

