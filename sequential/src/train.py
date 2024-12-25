import numpy as np
import torch
from tqdm import tqdm

import wandb
from src.models.ARattn import ARModel, CLIPCAModel
from src.models.MoEattn import MoEClipCA
from src.models.common import clip_loss
from src.utils import simple_ndcg_at_k_batch, simple_recall_at_k_batch


def train(
    model, optimizer, dataloader, criterion, alpha: float = 0.4, device: str = "cpu"
):
    model.train()
    total_loss = 0

    with tqdm(dataloader) as t:
        for tokens, labels, ori_emb, gen_emb, text_emb in t:
            tokens = tokens.to(device)
            ori_emb = ori_emb.to(device)
            gen_emb = gen_emb.to(device)
            labels = labels.to(device)

            if isinstance(model, MoEClipCA):
                enc_emb, logits  = model(tokens, ori_emb, text_emb)
                contra_loss = clip_loss(enc_emb, gen_emb, model.logit_scale)
            elif isinstance(model, CLIPCAModel):
                enc_emb, logits = model(tokens, ori_emb)
                contra_loss = clip_loss(enc_emb, gen_emb, model.logit_scale)
            elif isinstance(model, ARModel):
                logits = model(tokens, ori_emb, gen_emb)
            

            rec_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss = (
                (1 - alpha) * rec_loss + alpha * contra_loss
                if isinstance(model, CLIPCAModel)
                else rec_loss
            )
            t.set_postfix(loss=loss.item())
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "contra_loss": contra_loss.item(),
                    "rec_loss": rec_loss.item(),
                }
            )
            total_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / len(dataloader)


def eval(
    model,
    mode,
    dataloader,
    criterion,
    train_data,
    alpha: float = 0.4,
    device: str = "cpu",
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
            for users, tokens, labels, ori_emb, gen_emb, text_emb in t:
                tokens = tokens.to(device)
                ori_emb = ori_emb.to(device)
                gen_emb = gen_emb.to(device)
                labels = labels.to(device)

                if isinstance(model, CLIPCAModel):
                    enc_emb, logits = model(tokens, ori_emb)
                elif isinstance(model, (ARModel)):
                    logits = model(tokens, ori_emb, gen_emb)
                elif isinstance(model, MoEClipCA):
                    logits = model(tokens, ori_emb, gen_emb, text_emb)

                if mode == "valid":
                    rec_loss = criterion(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                    if isinstance(model, CLIPCAModel):
                        contra_loss = alpha * clip_loss(
                            enc_emb, gen_emb, model.logit_scale
                        )

                    loss = (
                        (1 - alpha) * rec_loss + alpha * contra_loss
                        if isinstance(model, CLIPCAModel)
                        else rec_loss
                    )
                    t.set_postfix(loss=loss.item())
                    wandb.log(
                        {
                            "valid_batch_loss": loss.item(),
                            "valid_contra_loss": contra_loss.item(),
                            "vallid_rec_loss": rec_loss.item(),
                        }
                    )
                    total_loss += loss

                ### GET METRICS ###

                used_items_batch = [
                    np.unique(train_data[user]) for user in users.cpu().numpy()
                ]
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
