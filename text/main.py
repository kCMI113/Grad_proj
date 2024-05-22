import pandas as pd
import torch
from dataset import TokenDataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from utils import compute_metrics, get_timestamp

PATH = "./data"


def main():
    data = pd.read_csv(f"{PATH}/aug6_no_unknown.csv")
    garment_group_dict = torch.load(f"{PATH}/garment_group_name_dict.pt")
    garment_group_dict_inv = torch.load(f"{PATH}/garment_group_dict_inv.pt")

    train_df, test_df = train_test_split(data, test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    train_dataset = TokenDataset(train_df, 256, tokenizer)
    test_dataset = TokenDataset(test_df, 256, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=len(garment_group_dict),
        id2label=garment_group_dict_inv,
        label2id=garment_group_dict,
    )

    training_args = TrainingArguments(
        output_dir=f"./seq_clf/{get_timestamp()}",
        learning_rate=2e-5,
        num_train_epochs=25,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        auto_find_batch_size=True,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
