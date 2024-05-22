import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, df, max_len, tokenizer) -> None:
        self.df = df
        self.df.detail_desc = self.df.detail_desc.astype(str)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx]["detail_desc"]
        label = self.df.iloc[idx]["label"]
        label_list = [0 for i in range(21)]
        label_list[label] = 1
        tokens = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_len,
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        torch.zeros_like(attention_mask)

        return {"input_ids": input_ids, "label": torch.tensor(label)}
