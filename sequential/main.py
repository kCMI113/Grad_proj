# flake8: noqa
import argparse
import os
import shutil

import dotenv
import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import snapshot_download, HfApi

from recbole.model.loss import BPRLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

import wandb
from src import dataset as DS
from src.models.ARattn import ARModel, CLIPCAModel
from src.models.MoEattn import MoEClipCA
from src.sampler import HardNegSampler, NegSampler
from src.train import eval, train
from src.utils import get_config, get_timestamp, load_json, mk_dir, seed_everything

torch.autograd.set_detect_anomaly(True)


def main(args):
    ############# SETTING #############
    setting_yaml_path = f"./settings/{args.config}.yaml"
    timestamp = get_timestamp()
    models = {"AR": ARModel, "CLIPCA": CLIPCAModel, "MoE": MoEClipCA}
    seed_everything()
    mk_dir("./model")
    mk_dir("./data")
    mk_dir(f"./model/{timestamp}")

    settings = get_config(setting_yaml_path)

    model_name: str = settings["model_name"]
    model_args: dict = settings["model_arguments"]
    model_dataset: dict = settings["model_dataset"]
    settings["experiment_name"] = (
        f"work-{timestamp}_{model_name}_"
        + f'{settings['batch_size']}_{model_args["hidden_size"]}_{model_args["num_attention_heads"]}_{model_args["num_hidden_layers"]}_{settings["lr"]}_{settings["weight_decay"]}'
        + (
            ""
            if model_name == "AR"
            else f'_{model_args["num_enc_layers"]}_{model_args["num_enc_heads"]}_{settings["alpha"]}_{settings["schedule_rate"]}'
        )
        + ("_useLinear" if model_args["use_linear"] else "")
        + ("_shuffle" if settings["shuffle"] else "")
    )

    shutil.copy(setting_yaml_path, f"./model/{timestamp}/setting.yaml")

    ############ SET HYPER PARAMS #############
    ## TRAIN ##
    lr = settings["lr"]
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
        name=settings["experiment_name"],
        mode=os.environ.get("WANDB_MODE"),
    )
    wandb.log(settings)
    wandb.save(setting_yaml_path)

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
        )
    else:
        path = f"./data/{dataset}"

    _parameter = {
        "max_len": model_args["max_len"],
    }

    print("-------------LOAD DATA-------------")
    metadata = load_json(f"{path}/20_core_metadata.json")
    # train_data = torch.load(f"./data/xs/train_data_new_xs.pt")
    test_data = torch.load(f"{path}/uniqued_test_data_xs.pt")
    valid_data = [v[:-1] for v in test_data]
    train_data = [v[:-2] for v in test_data]
    # _parameter["gen_img_emb"] = torch.load(f"./data/xs/gen_emb_new_dict.pt")
    _parameter["ori_img_emb"] = torch.load(f"{path}/idx2img_emb.pt")
    _parameter["text_emb"] = torch.load(f"{path}/idx2text_emb.pt")

    num_user = len(test_data)
    num_item = len(_parameter["ori_img_emb"])

    _parameter["num_user"] = num_user
    _parameter["num_item"] = num_item
    print(_parameter["num_user"], _parameter["num_item"])

    print("-------------COMPLETE LOAD DATA-------------")

    train_dataset_class_ = getattr(DS, model_dataset["train_dataset"])
    test_dataset_class_ = getattr(DS, model_dataset["test_dataset"])

    train_dataset = train_dataset_class_(
        user_seq=test_data,
        type="Train",
        **_parameter,
    )
    valid_dataset = test_dataset_class_(
        user_seq=test_data,
        type="Valid",
        **_parameter,
    )
    test_dataset = test_dataset_class_(
        user_seq=test_data,
        type="Test",
        **_parameter,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=settings["shuffle"],
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    ############# SETTING FOR TRAIN #############
    device = f"cuda:{n_cuda}" if n_cuda != "cpu" else "cpu"

    ## MODEL INIT ##
    model_class_ = models[model_name]

    model = model_class_(
        **model_args,
        num_item=num_item,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss = train(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            criterion=criterion,
            alpha=settings["alpha"],
            device=device,
        )
        print(
            f'EPOCH : {i+1} | TRAIN LOSS : {train_loss} | ALPHA : {settings["alpha"]}'
        )
        wandb.log({"loss": train_loss, "epoch": i + 1})

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
                alpha=settings["alpha"],
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

            print("-------------EVAL-------------")
            test_metrics = eval(
                model=model,
                mode="test",
                dataloader=test_dataloader,
                criterion=criterion,
                train_data=valid_data,
                device=device,
                alpha=settings["alpha"],
            )
            print(
                (
                    f'R1 : {test_metrics["R1"]} | R5 : {test_metrics["R5"]} | R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | R40 : {test_metrics["R40"]} | '
                    f'N1 : {test_metrics["N1"]} | N5 : {test_metrics["N5"]} | N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} | N40 : {test_metrics["N40"]}'
                )
            )
            test_metrics["epoch"] = i + 1
            wandb.log(test_metrics)
            settings["alpha"] = (
                settings["alpha"] - settings["schedule_rate"]
                if settings["alpha"] - settings["schedule_rate"] > 0.15
                else 0.15
            )  # update alpha
            wandb.log({"epoch": i + 1, "alpha": settings["alpha"]})

    print("-------------FINAL EVAL-------------")
    test_metrics = eval(
        model=model,
        mode="test",
        dataloader=test_dataloader,
        criterion=criterion,
        train_data=valid_data,
        device=device,
        alpha=settings["alpha"],
    )
    print(
        (
            f'R1 : {test_metrics["R1"]} | R5 : {test_metrics["R5"]} | R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | R40 : {test_metrics["R40"]} | '
            f'N1 : {test_metrics["N1"]} | N5 : {test_metrics["N5"]} | N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} | N40 : {test_metrics["N40"]}'
        )
    )
    print("#################### SAVE MODEL CHECKPOINT ####################")
    model_save_path = f"model/{timestamp}/{settings["experiment_name"]}"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(), f"{model_save_path}/final_weights.pt")
    wandb.save(f"model/{model_save_path}/final_weights.pt")
    # Upload to Huggingface Hub
    api = HfApi()
    api.upload_folder(
        folder_path=model_save_path,
        repo_id="SLKpnu/mmp_hm",
        path_in_repo=model_save_path.split("/")[1],
        commit_message=f"{model_name}_{model_save_path.split('/')[1]} | run_name : "
        + settings["experiment_name"],
        repo_type="model",
    )
    wandb.log(test_metrics)

    ############ WANDB FINISH #############
    wandb.finish()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        action="store",
        required=False,
        default="MoE",
    )
    args = parser.parse_args()
    main(args)
