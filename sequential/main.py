# flake8: noqa
import argparse
import os
import shutil

import dotenv
import torch
import torch.nn as nn
import wandb
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from recbole.model.loss import BPRLoss
from src import dataset as DS
from src.models.ARattn import ARModel
from src.models.bert import BERT4Rec
from src.models.crossattention import (
    CA4Rec,
    DOCA4Rec,
    DOCAdvanced,
    MMDOCA4Rec,
    MMDOCAdvanced,
)
from src.models.mlp import MLPRec
from src.models.mlpbert import MLPBERT4Rec
from src.sampler import HardNegSampler, NegSampler
from src.train import eval, train
from src.utils import get_config, get_timestamp, load_json, mk_dir, seed_everything
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader


def main(args):
    ############# SETTING #############
    setting_yaml_path = f"./settings/{args.config}.yaml"
    timestamp = get_timestamp()
    models = {
        "BERT4Rec": BERT4Rec,
        "MLPRec": MLPRec,
        "MLPBERT4Rec": MLPBERT4Rec,
        "CA4Rec": CA4Rec,
        "DOCA4Rec": DOCA4Rec,
        "MMDOCA4Rec": MMDOCA4Rec,
        "DOCAdvanced": DOCAdvanced,
        "MMDOCAdvanced": MMDOCAdvanced,
        "AR": ARModel,
    }
    seed_everything()
    mk_dir("./model")
    mk_dir("./data")
    mk_dir(f"./model/{timestamp}")

    settings = get_config(setting_yaml_path)

    model_name: str = settings["model_name"]
    model_args: dict = settings["model_arguments"]
    model_dataset: dict = settings["model_dataset"]
    name = f"work-{timestamp}_" + settings["experiment_name"]

    shutil.copy(setting_yaml_path, f"./model/{timestamp}/setting.yaml")

    ############ SET HYPER PARAMS #############
    ## TRAIN ##
    lr = settings["lr"]
    lr_step = settings["lr_step"]
    epoch = settings["epoch"]
    batch_size = settings["batch_size"]
    weight_decay = settings["weight_decay"]
    num_workers = settings["num_workers"]

    ## DATA ##
    data_local = settings["data_local"]
    data_repo = settings["data_repo"]
    dataset = settings["dataset"]
    data_version = settings["data_version"]

    ## ETC ##
    n_cuda = settings["n_cuda"]

    ############ WANDB INIT #############
    print("--------------- Wandb SETTING ---------------")
    dotenv.load_dotenv()
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        name=name,
        mode=os.environ.get("WANDB_MODE"),
    )
    wandb.log(settings)

    ############# LOAD DATASET #############
    # when calling data from huggingface Hub
    if not data_local:
        path = (
            snapshot_download(
                repo_id=f"SLKpnu/{data_repo}",
                repo_type="dataset",
                cache_dir="./data",
                revision=data_version,
            )
            + "/"
            + dataset
        )
    else:
        path = f"./data/{dataset}"

    _parameter = {
        "max_len": model_args["max_len"],
        "n_negs": model_args["n_HNS"] + model_args["n_NS"],
    }

    print("-------------LOAD DATA-------------")
    metadata = load_json(f"{path}/metadata.json")
    train_data = torch.load(f"./data/xs/train_data_new_xs.pt")
    test_data = torch.load(f"./data/xs/test_data_new_xs.pt")
    _parameter["gen_img_emb"] = torch.load(f"./data/xs/gen_emb_new_dict.pt")
    _parameter["ori_img_emb"] = torch.load(f"./data/xs/ori_emb_new_dict.pt")

    num_user = len(train_data)
    num_item = len(_parameter["gen_img_emb"])

    _parameter["num_user"] = num_user
    _parameter["num_item"] = num_item

    print("-------------COMPLETE LOAD DATA-------------")

    train_dataset_class_ = getattr(DS, model_dataset["train_dataset"])
    test_dataset_class_ = getattr(DS, model_dataset["test_dataset"])

    # _parameter["gen_img_emb"] = torch.load("./noise_img_emb_09.pt").type(torch.float)
    _parameter["txt_emb"] = torch.load(f"{path}/detail_text_embeddings.pt")
    _parameter["gen_img_idx"] = torch.load(f"{path}/input_gen_img.pt")
    # _parameter["mean"] = model_args["mean"]
    # _parameter["std"] = model_args["std"]

    hnsampler, negsampler = None, None
    # conditional DATA
    if model_args["loss"] == "BPR":
        if model_args["n_HNS"] > 0:
            sim_item_mat = torch.load(f"{path}/gen_img_sim_item_int32.pt")
            hnsampler = HardNegSampler(model_args["n_HNS"], sim_item_mat)
        if model_args["n_NS"] > 0:
            negsampler = NegSampler(model_args["n_NS"], num_item)

    train_dataset = train_dataset_class_(
        user_seq=test_data,
        hnsampler=hnsampler,
        negsampler=negsampler,
        mask_prob=model_args["mask_prob"],
        type="Train",
        **_parameter,
    )
    valid_dataset = test_dataset_class_(
        user_seq=test_data,
        hnsampler=hnsampler,
        negsampler=negsampler,
        type="Valid",
        **_parameter,
    )
    test_dataset = test_dataset_class_(
        user_seq=test_data,
        hnsampler=hnsampler,
        negsampler=negsampler,
        type="Test",
        **_parameter,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    ############# SETTING FOR TRAIN #############
    device = f"cuda:{n_cuda}" if torch.cuda.is_available() else "cpu"

    ## MODEL INIT ##
    model_class_ = models[model_name]

    if model_name in ("MLPBERT4Rec", "MLPRec", "MLPwithBERTFreeze"):
        model_args["linear_in_size"] = model_args["hidden_size"]

    model = model_class_(
        **model_args,
        num_item=num_item,
        device=device,
    ).to(device)

    criterion = (
        nn.CrossEntropyLoss(ignore_index=0) if model_args["loss"] == "CE" else BPRLoss()
    )

    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step, gamma=0.5)

    accelerator = Accelerator()
    device = accelerator.device
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss = train(
            accelerator,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            criterion,
            device,
        )
        print(
            f'EPOCH : {i+1} | TRAIN LOSS : {train_loss} | LR : {optimizer.param_groups[0]["lr"]}'
        )
        wandb.log(
            {"loss": train_loss, "epoch": i + 1, "lr": optimizer.param_groups[0]["lr"]}
        )

        if i % settings["valid_step"] == 0:
            print("-------------VALID-------------")
            (
                valid_loss,
                valid_metrics,
            ) = eval(
                model=model,
                mode="valid",
                dataloader=valid_dataloader,
                criterion=criterion,
                train_data=train_data,
                device=device,
            )
            print(f"EPOCH : {i+1} | VALID LOSS : {valid_loss}")
            print(
                (
                    f'R1 : {valid_metrics["R1"]} | R5 : {valid_metrics["R5"]} | R10 : {valid_metrics["R10"]} | R20 : {valid_metrics["R20"]} | R40 : {valid_metrics["R40"]} | '
                    f'N1 : {valid_metrics["N1"]} | N5 : {valid_metrics["N5"]} | N10 : {valid_metrics["N10"]} | N20 : {valid_metrics["N20"]} | N40 : {valid_metrics["N40"]}'
                )
            )
            wandb.log(
                {
                    "epoch": i + 1,
                    "valid_loss": valid_loss,
                    "valid_R1": valid_metrics["R1"],
                    "valid_R5": valid_metrics["R5"],
                    "valid_R10": valid_metrics["R10"],
                    "valid_R20": valid_metrics["R20"],
                    "valid_R40": valid_metrics["R40"],
                    "valid_N1": valid_metrics["N1"],
                    "valid_N5": valid_metrics["N5"],
                    "valid_N10": valid_metrics["N10"],
                    "valid_N20": valid_metrics["N20"],
                    "valid_N40": valid_metrics["N40"],
                }
            )
            # torch.save(
            #     model.state_dict(), f"./model/{timestamp}/model_val_{valid_loss}.pt"
            # )

    print("-------------EVAL-------------")
    test_metrics = eval(
        model=model,
        mode="test",
        dataloader=test_dataloader,
        criterion=criterion,
        train_data=train_data,
        device=device,
    )
    print(
        (
            f'R1 : {test_metrics["R1"]} | N5 : {test_metrics["N5"]} | R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | R40 : {test_metrics["R40"]} | '
            f'N1 : {test_metrics["N1"]} | N5 : {test_metrics["N5"]} | N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} | N40 : {test_metrics["N40"]}'
        )
    )
    wandb.log(test_metrics)
    # torch.save(pred_list, f"./model/{timestamp}/prediction.pt")
    # wandb.save(f"./model/{timestamp}/prediction.pt")

    ############ WANDB FINISH #############
    wandb.finish()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", dest="config", action="store", required=False, default="AR"
    )
    args = parser.parse_args()
    main(args)
