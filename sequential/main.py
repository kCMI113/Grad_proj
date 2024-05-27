# flake8: noqa
import os
import shutil

import dotenv
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from src import dataset as DS
from src.custom_optimizer import MultiOptimizer, MultiScheduler
from src.models.bert import BERT4Rec
from src.models.crossattention import CA4Rec, DOCA4Rec
from src.models.mlp import MLPRec
from src.models.mlpbert import MLPBERT4Rec
from src.train import eval, train
from src.utils import get_config, get_timestamp, load_json, mk_dir, seed_everything
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

import wandb


def main():
    ############# SETTING #############
    setting_yaml_path = "./settings/base.yaml"
    timestamp = get_timestamp()
    models = {
        "BERT4Rec": BERT4Rec,
        "MLPRec": MLPRec,
        "MLPBERT4Rec": MLPBERT4Rec,
        "CA4Rec": CA4Rec,
        "DOCA4Rec": DOCA4Rec,
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
    lr_milestones = settings["lr_milestones"]
    lr_encoder_gamma = settings["lr_encoder_gamma"]
    lr_decoder_gamma = settings["lr_decoder_gamma"]
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

    print("-------------LOAD DATA-------------")
    metadata = load_json(f"{path}/metadata.json")
    train_data = torch.load(f"{path}/train_data.pt")
    valid_data = torch.load(f"{path}/valid_data.pt")
    test_data = torch.load(f"{path}/test_data.pt")

    # conditional DATA
    # negative sampling
    torch.load(f"{path}/sim_matrix_sorted.pt") if model_args["neg_sampling"] else None

    # input is text embeddings grouped by description
    if model_args["detail_text"]:
        text_emb = torch.load(f"{path}/detail_text_embeddings.pt")
        if model_args["std"] < 0:
            gen_emb = torch.load(f"{path}/gen_img_emb.pt")
            gen_emb = gen_emb.reshape((-1, 512))
            gen_std = torch.std(gen_emb, dim=0)

    # input is generative and origin image grouped by description
    if model_args["gen_img"]:
        gen_img_emb = torch.load(f"{path}/item_idx_gen_embs.pt")
        origin_img_emb = torch.load(f"{path}/origin_img_emb.pt")

    num_user = metadata["num of user"]
    num_item = metadata["num of item"]

    print("-------------COMPLETE LOAD DATA-------------")

    _parameter = {
        "num_user": num_user,
        "num_item": num_item,
        "max_len": model_args["max_len"],
        "mask_prob": model_args["mask_prob"],
    }

    train_dataset_class_ = getattr(DS, model_dataset["train_dataset"])
    test_dataset_class_ = getattr(DS, model_dataset["test_dataset"])

    if model_args["gen_img"]:
        _parameter["origin_img_emb"] = origin_img_emb
        _parameter["gen_img_emb"] = gen_img_emb
        _parameter["closest_origin"] = model_args["closest_origin"]

        train_dataset = train_dataset_class_(user_seq=train_data, **_parameter)
        valid_dataset = test_dataset_class_(user_seq=valid_data, **_parameter)
        test_dataset = test_dataset_class_(user_seq=test_data, **_parameter)

    elif model_args["detail_text"]:
        _parameter["text_emb"] = text_emb
        _parameter["mean"] = model_args["mean"]
        _parameter["std"] = gen_std if model_args["std"] < 0 else model_args["std"]

        train_dataset = train_dataset_class_(user_seq=train_data, **_parameter)
        valid_dataset = test_dataset_class_(user_seq=valid_data, **_parameter)
        test_dataset = test_dataset_class_(user_seq=test_data, **_parameter)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

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

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if model_name == "CA4Rec":
        optimizer = MultiOptimizer(model, lr, weight_decay)
        scheduler = MultiScheduler(
            optimizer.encoder_optimizer,
            optimizer.decoder_optimizer,
            milestones=lr_milestones,
            gamma1=lr_encoder_gamma,
            gamma2=lr_decoder_gamma,
        )
    else:
        optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step, gamma=0.5)

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss = train(model, optimizer, scheduler, train_dataloader, criterion, device)
        if model_name == "CA4Rec":
            print(
                f'EPOCH : {i+1} | TRAIN LOSS : {train_loss} | LR : {optimizer.param_groups["lr_encoder"]} | {optimizer.param_groups["lr_decoder"]}'
            )
            wandb.log(
                {
                    "loss": train_loss,
                    "epoch": i + 1,
                    "lr_encoder": optimizer.param_groups["lr_encoder"],
                    "lr_decoder": optimizer.param_groups["lr_decoder"],
                }
            )
        else:
            print(f'EPOCH : {i+1} | TRAIN LOSS : {train_loss} | LR : {optimizer.param_groups[0]["lr"]}')
            wandb.log({"loss": train_loss, "epoch": i + 1, "lr": optimizer.param_groups[0]["lr"]})

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
                    f'R1 : {valid_metrics["R1"]} | R10 : {valid_metrics["R10"]} | R20 : {valid_metrics["R20"]} | R40 : {valid_metrics["R40"]} | '
                    f'N1 : {valid_metrics["N1"]} | N10 : {valid_metrics["N10"]} | N20 : {valid_metrics["N20"]} | N40 : {valid_metrics["N40"]}'
                )
            )
            wandb.log(
                {
                    "epoch": i + 1,
                    "valid_loss": valid_loss,
                    "valid_R1": valid_metrics["R1"],
                    "valid_R10": valid_metrics["R10"],
                    "valid_R20": valid_metrics["R20"],
                    "valid_R40": valid_metrics["R40"],
                    "valid_N1": valid_metrics["N1"],
                    "valid_N10": valid_metrics["N10"],
                    "valid_N20": valid_metrics["N20"],
                    "valid_N40": valid_metrics["N40"],
                }
            )
            torch.save(model.state_dict(), f"./model/{timestamp}/model_val_{valid_loss}.pt")

    print("-------------EVAL-------------")
    pred_list, test_metrics = eval(
        model=model,
        mode="test",
        dataloader=test_dataloader,
        criterion=criterion,
        train_data=train_data,
        device=device,
    )
    print(
        (
            f'R1 : {valid_metrics["R1"]} | R10 : {valid_metrics["R10"]} | R20 : {valid_metrics["R20"]} | R40 : {valid_metrics["R40"]} | '
            f'N1 : {valid_metrics["N1"]} | N10 : {valid_metrics["N10"]} | N20 : {valid_metrics["N20"]} | N40 : {valid_metrics["N40"]}'
        )
    )
    wandb.log(test_metrics)
    torch.save(pred_list, f"./model/{timestamp}/prediction.pt")
    wandb.save(f"./model/{timestamp}/prediction.pt")

    ############ WANDB FINISH #############
    wandb.finish()


if __name__ == "__main__":
    main()
