import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.models.ARattn import ARModel, CLIPCAModel
from src.models.MoEattn import MoEClipCA
from src.models.TMoEattn import (
    TMoEClipCA,
    TMoEClipCA_lienar,
    TMoEClipCA_M,
    TMoEClipCA_C,
)
from src.models.common import clip_loss
from src.utils import simple_ndcg_at_k_batch, simple_recall_at_k_batch
from src.models.SASRec import SASRec


def train(
    model,
    optimizer,
    dataloader,
    criterion,
    scheduler,
    alpha: float = 0.4,
    beta: float = None,
    theta: float = None,
    rec_weight: float = None,
    device: str = "cpu",
    epoch: int = None,
    loss_threshold: float = 0.2,
):
    model.train()
    total_loss = 0

    with tqdm(dataloader) as t:
        for tokens, labels, *args in t:
            tokens = tokens.to(device)
            labels = labels.to(device)

            if isinstance(model, CLIPCAModel):
                ori_emb, gen_emb, text_emb, prompt_emb = args
                ori_emb = ori_emb.to(device)
                gen_emb = gen_emb.to(device)
                text_emb = text_emb.to(device)
                prompt_emb = prompt_emb.to(device)

            if isinstance(model, (TMoEClipCA_C)):
                gen_res, prompt_res, gen_mm, item_emb, logits, gate_mean = model(
                    tokens, ori_emb, text_emb
                )
                img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)
                cosine_loss = 1 - torch.cosine_similarity(
                    gen_mm, item_emb[labels], dim=-1
                )
                mask = labels != 0
                cosine_loss = cosine_loss[mask].mean()
                contra_loss = alpha * img_loss + beta * text_loss + theta * cosine_loss
                wandb.log(
                    {
                        "img_loss": img_loss.item(),
                        "text_loss": text_loss.item(),
                        "cosine_loss": cosine_loss.item(),
                        "gate_mean": gate_mean.item(),
                    }
                )
                # rec_w = 1 - alpha - beta - theta
            elif isinstance(model, (TMoEClipCA, TMoEClipCA_lienar, TMoEClipCA_M)):
                gen_res, prompt_res, logits, gate_mean = model(
                    tokens, ori_emb, text_emb
                )
                img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)
                contra_loss = alpha * img_loss + beta * text_loss
                wandb.log(
                    {
                        "img_loss": img_loss.item(),
                        "text_loss": text_loss.item(),
                        "gate_mean": gate_mean.item(),
                    }
                )
                # rec_w = 1 - alpha - beta
            elif isinstance(model, MoEClipCA):
                enc_emb, logits = model(tokens, ori_emb, text_emb)
                img_loss = clip_loss(enc_emb, gen_emb, model.logit_scale)
                wandb.log(
                    {
                        "img_loss": img_loss.item(),
                    }
                )
                contra_loss = alpha * img_loss
                # rec_w = 1 - alpha
            elif isinstance(model, CLIPCAModel):
                enc_emb, logits = model(tokens, ori_emb)
                img_loss = clip_loss(enc_emb, gen_emb, model.logit_scale)
                wandb.log(
                    {
                        "img_loss": img_loss.item(),
                    }
                )
                contra_loss = alpha * img_loss
                # rec_w = 1 - alpha
            elif isinstance(model, ARModel):
                logits = model(tokens, ori_emb, gen_emb)
            elif isinstance(model, SASRec):
                logits = model(tokens)

            rec_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # loss = (
            #     rec_loss + contra_loss if isinstance(model, CLIPCAModel) else rec_loss
            # )
            if isinstance(model, CLIPCAModel):
                loss = (
                    # (rec_w * rec_loss + contra_loss)
                    (rec_weight * rec_loss + contra_loss)
                    if alpha <= loss_threshold
                    else contra_loss
                )
                wandb.log(
                    {
                        "contra_loss": contra_loss.item(),
                    }
                )
            else:
                loss = rec_loss

            t.set_postfix(
                {"loss": loss.item(), "gm": gate_mean.item()}
                if isinstance(
                    model, (TMoEClipCA, TMoEClipCA_lienar, TMoEClipCA_M, TMoEClipCA_C)
                )
                else {"loss": loss.item()}
            )
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                }
            )
            total_loss += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

    scheduler.step()
    return total_loss / len(dataloader)


def eval(
    model,
    mode,
    dataloader,
    criterion,
    train_data,
    alpha: float = 0.4,
    beta: float = None,
    theta: float = None,
    rec_weight: float = None,
    device: str = "cpu",
    loss_threshold: float = 0.2,
):
    model.eval()
    metrics_batches = {
        k: torch.tensor([]).to(device)
        for k in ["R1", "R5", "R10", "R20", "R40", "N1", "N5", "N10", "N20", "N40"]
    }
    total_loss = 0
    pred_list = []

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for _, tokens, labels, *args in t:
                tokens = tokens.to(device)
                labels = labels.to(device)

            if isinstance(model, CLIPCAModel):
                ori_emb, gen_emb, text_emb, prompt_emb = args
                ori_emb = ori_emb.to(device)
                gen_emb = gen_emb.to(device)
                text_emb = text_emb.to(device)
                prompt_emb = prompt_emb.to(device)

            if isinstance(model, (TMoEClipCA_C)):
                gen_res, prompt_res, gen_mm, item_emb, logits, gate_mean = model(
                    tokens, ori_emb, text_emb
                )
                if mode == "valid":
                    img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                    text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)
                    cosine_loss = 1 - torch.cosine_similarity(
                        gen_mm, item_emb[labels], dim=-1
                    )
                    mask = labels != 0
                    cosine_loss = cosine_loss[mask].mean()
                    contra_loss = (
                        alpha * img_loss + beta * text_loss + theta * cosine_loss
                    )
                    wandb.log(
                        {
                            "valid_img_loss": img_loss.item(),
                            "valid_text_loss": text_loss.item(),
                            "valid_cosine_loss": cosine_loss.item(),
                            "valid_gate_mean": gate_mean.item(),
                        }
                    )
                    # rec_w = 1 - alpha - beta - theta

            elif isinstance(model, (TMoEClipCA, TMoEClipCA_lienar, TMoEClipCA_M)):
                gen_res, prompt_res, logits, gate_mean = model(
                    tokens, ori_emb, text_emb
                )
                if mode == "valid":
                    img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                    text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)
                    contra_loss = alpha * img_loss + beta * text_loss
                    wandb.log(
                        {
                            "valid_img_loss": img_loss.item(),
                            "valid_text_loss": text_loss.item(),
                            "valid_gate_mean": gate_mean.item(),
                        }
                    )
                    # rec_w = 1 - alpha - beta
            elif isinstance(model, MoEClipCA):
                enc_emb, logits = model(tokens, ori_emb, text_emb)
                contra_loss = (
                    alpha * clip_loss(enc_emb, gen_emb, model.logit_scale)
                    if mode == "valid"
                    else None
                )
                # rec_w = 1 - alpha

            elif isinstance(model, CLIPCAModel):
                enc_emb, logits = model(tokens, ori_emb)
                contra_loss = (
                    alpha * clip_loss(enc_emb, gen_emb, model.logit_scale)
                    if mode == "valid"
                    else None
                )
                # rec_w = 1 - alpha

            elif isinstance(model, ARModel):
                logits = model(tokens, ori_emb, gen_emb)
            elif isinstance(model, SASRec):
                logits = model(tokens)

            if mode == "valid":
                rec_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                if isinstance(model, CLIPCAModel):
                    loss = (
                        # rec_w * rec_loss + contra_loss
                        (rec_weight * rec_loss + contra_loss)
                        if alpha <= loss_threshold
                        else contra_loss
                    )
                    wandb.log(
                        {
                            "valid_contra_loss": contra_loss.item(),
                        }
                    )
                else:
                    loss = rec_loss

                t.set_postfix(loss=loss.item())
                wandb.log(
                    {
                        "valid_batch_loss": loss.item(),
                        "valid_rec_loss": rec_loss.item(),
                    }
                )

                total_loss += loss

            ### GET METRICS ###

            # used_items_batch = [
            #     np.unique(train_data[user]) for user in users.cpu().numpy()
            # ]
            target_batch = labels[:, -1]
            user_res_batch = -logits[:, -1, 1:]

            # for i, used_item_list in enumerate(used_items_batch):
            #     user_res_batch[i][used_item_list] = user_res_batch[i].max() + 1

            # sorted item id e.g. [3452(1st), 7729(2nd), ... ]
            item_rank_batch = user_res_batch.argsort()

            # if mode == "test":
            #     pred_list.append(
            #         torch.concat(
            #             (item_rank_batch[:, :40], target_batch.unsqueeze(1)), dim=1
            #         )
            #     )

            # rank of items e.g. index: item_id(0~), item_rank[0] : rank of item_id 0
            item_rank_batch = (
                item_rank_batch.argsort()
                .gather(dim=1, index=target_batch.view(-1, 1) - 1)
                .squeeze()
            )

            for k in [1, 5, 10, 20, 40]:
                recall = simple_recall_at_k_batch(k, item_rank_batch)
                ndcg = simple_ndcg_at_k_batch(k, item_rank_batch)

                metrics_batches["R" + str(k)] = torch.cat(
                    (metrics_batches["R" + str(k)], recall)
                )
                metrics_batches["N" + str(k)] = torch.cat(
                    (metrics_batches["N" + str(k)], ndcg)
                )

        for k in [1, 5, 10, 20, 40]:
            metrics_batches["R" + str(k)] = metrics_batches["R" + str(k)].mean()
            metrics_batches["N" + str(k)] = metrics_batches["N" + str(k)].mean()

        if mode == "valid":
            return total_loss / len(dataloader), metrics_batches

    # pred_list = torch.cat(pred_list).tolist()
    return metrics_batches
