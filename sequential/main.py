# flake8: noqa
import argparse
import os
import shutil

import dotenv
import torch
import torch.nn as nn
import wandb

# from accelerate import Accelerator
from huggingface_hub import HfApi, snapshot_download
from src import dataset as DS
from src.models.ARattn import ARModel, CLIPCAModel
from src.models.common import EarlyStopping
from src.models.MoEattn import MoEClipCA
from src.models.SASRec import SASRec
from src.models.TMoEattn import TMoEClipCA, TMoEClipCA_C, TMoEClipCA_CO
from src.train import eval, train
from src.utils import get_config, get_timestamp, load_json, mk_dir, seed_everything

# from recbole.model.loss import BPRLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

# torch.autograd.set_detect_anomaly(True)


def main(settings):
    ############# SETTING #############
    timestamp = get_timestamp()
    models = {
        "AR": ARModel,
        "CLIPCA": CLIPCAModel,
        "MoE": MoEClipCA,
        "TMoE": TMoEClipCA,
        "TMoEC": TMoEClipCA_C,
        "TMoECO": TMoEClipCA_CO,
        "SASRec": SASRec,
    }
    seed_everything()
    mk_dir("./model")
    mk_dir("./data")
    mk_dir(f"./model/{timestamp}")

    model_name: str = settings["model_name"]
    model_args: dict = settings["model_arguments"]
    model_dataset: dict = settings["model_dataset"]

    settings["experiment_name"] = (
        f"work-{timestamp}_{model_name}_"
        + f'{settings["batch_size"]}_{model_args["hidden_size"]}_{model_args["num_attention_heads"]}_{model_args["num_hidden_layers"]}_{settings["lr"]}_{settings["weight_decay"]}'
        + (
            f'_{model_args["num_enc_layers"]}_{model_args["num_enc_heads"]}_{settings["loss_threshold"]}_({settings["alpha"]}|{settings["alpha_threshold"]})_({settings["rec_weight"]})_{settings["schedule_rate"]}'
            if model_name not in ["SASRec"]
            else ""
        )
        + (
            f"_SA"
            if model_name in ["TMoE", "TMoEC", "TMoECO"]
            and model_args["num_gen_heads"] > 0
            else ("_MLP" if model_name not in ["SASRec"] else "")
        )
        + (
            f"_Sel"
            if model_name in ["TMoE", "TMoEC", "TMoECO"]
            and model_args["selected_modal_in"]
            else ""
        )
        + (
            f"_({settings['beta']}|{settings['beta_threshold']})"
            if model_name in ["TMoE", "TMoEC", "TMoECO"]
            else ""
        )
        + (f"_({settings['theta']})" if model_name in ["TMoEC", "TMoECO"] else "")
        + (
            "_useLinear"
            if model_name not in ["SASRec"] and model_args["use_linear"]
            else ""
        )
        + ("_shuffle" if settings["shuffle"] else "")
        + (
            "_modal_gate"
            if model_name in ["MoE", "TMoE", "TMoEC", "TMoECO"]
            and model_args["modal_gate"]
            else ""
        )
        + (
            f"_{model_args['num_experts']}"
            if model_name in ["MoE", "TMoE", "TMoEC", "TMoECO"]
            and model_args["num_experts"]
            else ""
        )
        + (
            "_" + model_args["text_model"]
            if model_name not in ["SASRec"] and model_args["text_model"] != "fclip"
            else ""
        )
        + (
            "_" + model_args["img_model"]
            if model_name not in ["SASRec"] and model_args["img_model"] != "fclip"
            else ""
        )
    )
    # shutil.copy(setting_yaml_path, f"./model/{timestamp}/setting.yaml")

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
    # wandb.save(setting_yaml_path)

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
    metadata = load_json(f"{path}/uniqued_metadata.json")
    # train_data = torch.load(f"./data/xs/train_data_new_xs.pt")
    test_data = torch.load(f"{path}/uniqued_test_data.pt")
    # _parameter["gen_img_emb"] = torch.load(f"./data/xs/gen_emb_new_dict.pt")
    # _parameter["img_emb"] = torch.load(f"{path}/idx_img_emb_map.pt")
    # _parameter["text_emb"] = torch.load(f"{path}/idx_text_emb_map.pt")
    if model_name not in ["SASRec"]:
        _parameter["img_emb"] = torch.load(
            f"{path}/idx_img_emb_map{'_'+model_args['img_model'] if model_args['img_model'] != 'fclip' else ''}.pt"
        )
        _parameter["text_emb"] = torch.load(
            f"{path}/idx_text_emb_map{'_'+model_args['text_model'] if model_args['text_model'] != 'fclip' else ''}.pt"
        )
        model_args["text_emb_size"] = _parameter["text_emb"][0].shape[-1]
        model_args["img_emb_size"] = _parameter["img_emb"][0].shape[-1]

    _parameter["num_user"] = metadata["num of user"]
    _parameter["num_item"] = metadata["num of item"]
    model_args["num_user"] = metadata["num of user"]
    model_args["num_item"] = metadata["num of item"]

    print(_parameter["num_user"], _parameter["num_item"])

    print("-------------COMPLETE LOAD DATA-------------")

    train_dataset_class_ = getattr(DS, f"{model_dataset}TrainDataset")
    valid_dataset_class_ = getattr(DS, f"{model_dataset}ValidDataset")
    test_dataset_class_ = getattr(DS, f"{model_dataset}TestDataset")

    train_dataset = train_dataset_class_(
        user_seq=test_data,
        **_parameter,
    )
    valid_dataset = valid_dataset_class_(
        user_seq=test_data,
        **_parameter,
    )
    test_dataset = test_dataset_class_(
        user_seq=test_data,
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
        # num_item=_parameter["num_item"],
        device=device,
    ).to(device)

    scheduler = None
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, delta=0)

    if settings["scheduler"] == "LambdaLR":
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: settings["scheduler_rate"] ** epoch,
        )

    ############# TRAIN AND EVAL #############
    for i in range(epoch):
        print("-------------TRAIN-------------")
        train_loss = train(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            criterion=criterion,
            alpha=settings["alpha"] if isinstance(model, CLIPCAModel) else None,
            beta=(
                settings["beta"] if model_name in ["TMoE", "TMoEC", "TMoECO"] else None
            ),
            rec_weight=settings["rec_weight"] if model_name not in ["SASRec"] else None,
            theta=(settings["theta"] if model_name in ["TMoEC", "TMoECO"] else None),
            device=device,
            epoch=i,
            loss_threshold=(
                settings["loss_threshold"] if isinstance(model, CLIPCAModel) else None
            ),
            scheduler=scheduler,
        )
        print(
            f"EPOCH : {i+1} | TRAIN LOSS : {train_loss} | LR : {optimizer.param_groups[0]['lr']}"
            + (
                f'| ALPHA : {settings["alpha"]}'
                if isinstance(model, CLIPCAModel)
                else ""
            )
        )
        wandb.log(
            {"loss": train_loss, "epoch": i + 1, "LR": optimizer.param_groups[0]["lr"]}
        )

        if i % settings["valid_step"] == 0:
            print("-------------VALID-------------")
            (
                # valid_loss,
                valid_metrics
            ) = eval(
                model=model,
                mode="valid",
                dataloader=valid_dataloader,
                criterion=criterion,
                alpha=settings["alpha"] if isinstance(model, CLIPCAModel) else None,
                beta=(
                    settings["beta"]
                    if model_name in ["TMoE", "TMoEC", "TMoECO"]
                    else None
                ),
                rec_weight=(
                    settings["rec_weight"] if model_name not in ["SASRec"] else None
                ),
                theta=(
                    settings["theta"] if model_name in ["TMoEC", "TMoECO"] else None
                ),
                device=device,
                loss_threshold=(
                    settings["loss_threshold"]
                    if isinstance(model, CLIPCAModel)
                    else None
                ),
            )
            # print(f"EPOCH : {i+1} | VALID LOSS : {valid_loss}")
            print(
                (
                    f'R1 : {valid_metrics["R1"]} | R5 : {valid_metrics["R5"]} | R10 : {valid_metrics["R10"]} | R20 : {valid_metrics["R20"]} | R40 : {valid_metrics["R40"]} | '
                    f'N1 : {valid_metrics["N1"]} | N5 : {valid_metrics["N5"]} | N10 : {valid_metrics["N10"]} | N20 : {valid_metrics["N20"]} | N40 : {valid_metrics["N40"]}'
                )
            )
            wandb.log(
                {
                    "epoch": i + 1,
                    # "valid_loss": valid_loss,
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
                    "valid_M1": valid_metrics["M1"],
                    "valid_M5": valid_metrics["M5"],
                    "valid_M10": valid_metrics["M10"],
                    "valid_M20": valid_metrics["M20"],
                    "valid_M40": valid_metrics["M40"],
                }
            )

            print("-------------EVAL-------------")
            test_metrics = eval(
                model=model,
                mode="test",
                dataloader=test_dataloader,
                criterion=criterion,
                device=device,
                alpha=settings["alpha"] if isinstance(model, CLIPCAModel) else None,
                beta=(
                    settings["beta"]
                    if model_name in ["TMoE", "TMoEC", "TMoECO"]
                    else None
                ),
                theta=(
                    settings["theta"] if model_name in ["TMoEC", "TMoECO"] else None
                ),
                rec_weight=(
                    settings["rec_weight"] if model_name not in ["SASRec"] else None
                ),
                loss_threshold=(
                    settings["loss_threshold"]
                    if isinstance(model, CLIPCAModel)
                    else None
                ),
            )
            print(
                (
                    f'R1 : {test_metrics["R1"]} | R5 : {test_metrics["R5"]} | R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | R40 : {test_metrics["R40"]} | '
                    f'N1 : {test_metrics["N1"]} | N5 : {test_metrics["N5"]} | N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} | N40 : {test_metrics["N40"]}'
                )
            )
            test_metrics["epoch"] = i + 1
            wandb.log(test_metrics)

            if isinstance(model, CLIPCAModel):
                settings["alpha"] = (
                    (settings["alpha"] - settings["schedule_rate"])
                    if (settings["alpha"] - settings["schedule_rate"])
                    > settings["alpha_threshold"]
                    else settings["alpha_threshold"]
                )  # update alpha
                wandb.log({"epoch": i + 1, "alpha": settings["alpha"]})

            if model_name in ["TMoE", "TMoEC", "TMoECO"]:
                settings["beta"] = (
                    (settings["beta"] - settings["schedule_rate"])
                    if (settings["beta"] - settings["schedule_rate"])
                    > settings["beta_threshold"]
                    else settings["beta_threshold"]
                )  # update beta
                wandb.log({"epoch": i + 1, "beta": settings["beta"]})

            if model_name in ["TMoEC", "TMoECO"]:
                settings["theta"] = (
                    (settings["theta"] - settings["schedule_rate"])
                    if (settings["theta"] - settings["schedule_rate"])
                    > settings["theta_threshold"]
                    else settings["theta_threshold"]
                )  # update beta
                wandb.log({"epoch": i + 1, "theta": settings["theta"]})

            if early_stopping(valid_metrics["R10"]):
                print(f"\033[43mEARLY STOPPED!!\033[0m \ntriggered at epoch {i + 1}")
                break
            else:
                print(
                    f"\033[43mKEEP GOING!!\033[0m \nEpoch {i + 1}: recall@10 = {valid_metrics['R10']}"
                )

    print("-------------FINAL EVAL-------------")
    test_metrics = eval(
        model=model,
        mode="test",
        dataloader=test_dataloader,
        criterion=criterion,
        device=device,
        alpha=settings["alpha"] if isinstance(model, CLIPCAModel) else None,
        beta=(settings["beta"] if model_name in ["TMoE", "TMoEC", "TMoECO"] else None),
        theta=(settings["theta"] if model_name in ["TMoEC", "TMoECO"] else None),
        loss_threshold=(
            settings["loss_threshold"] if isinstance(model, CLIPCAModel) else None
        ),
    )
    print(
        (
            f'R1 : {test_metrics["R1"]} | R5 : {test_metrics["R5"]} | R10 : {test_metrics["R10"]} | R20 : {test_metrics["R20"]} | R40 : {test_metrics["R40"]} | '
            f'N1 : {test_metrics["N1"]} | N5 : {test_metrics["N5"]} | N10 : {test_metrics["N10"]} | N20 : {test_metrics["N20"]} | N40 : {test_metrics["N40"]}'
        )
    )
    # print("#################### SAVE MODEL CHECKPOINT ####################")
    # model_save_path = f"model/{timestamp}/{settings['experiment_name']}"
    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)
    # torch.save(model.state_dict(), f"{model_save_path}/final_weights.pt")
    # wandb.save(f"model/{model_save_path}/final_weights.pt")
    # # Upload to Huggingface Hub

    # api = HfApi()
    # api.upload_folder(
    #     folder_path=model_save_path,
    #     repo_id="SLKpnu/mmp_fashion",
    #     path_in_repo=model_save_path.split("/")[1],
    #     commit_message=f"{model_name}_{model_save_path.split('/')[1]} | run_name : "
    #     + settings["experiment_name"],
    #     repo_type="model",
    # )
    # wandb.log(test_metrics)

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
        default="TMoE",
    )
    args = parser.parse_args()
    setting_yaml_path = f"./settings/{args.config}.yaml"
    settings = get_config(setting_yaml_path)

    main(settings)
