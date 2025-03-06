import numpy as np
import torch
import wandb
from src.models.ARattn import ARModel, CLIPCAModel
from src.models.common import clip_loss
from src.models.MoEattn import MoEClipCA
from src.models.SASRec import SASRec
from src.models.TMoEattn import TMoEClipCA, TMoEClipCA_C, TMoEClipCA_CO
from src.utils import AverageMeterSet, modi_absolute_recall_mrr_ndcg_for_ks
from tqdm import tqdm


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

            if isinstance(model, (TMoEClipCA_C)):  # 생성임베딩 - 타겟아이템 코사로스
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
            elif isinstance(model, (TMoEClipCA_CO)):  # 기존 모달 임베딩 <-> 아이템
                gen_res, prompt_res, topk, item_emb, logits, gate_mean = model(
                    tokens, ori_emb, text_emb
                )
                img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)

                topk = topk.unsqueeze(-1)
                mm_info = torch.stack((gen_emb, prompt_emb), dim=-2)
                mm_info = torch.gather(
                    mm_info, dim=2, index=topk.expand(-1, -1, -1, mm_info.size(-1))
                ).squeeze(-2)

                cosine_loss = 1 - torch.cosine_similarity(
                    mm_info, item_emb[labels], dim=-1
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
                        "a_gate_mean": torch.mean(topk.to(torch.float32)).item(),
                    }
                )
                # rec_w = 1 - alpha - beta - theta
            elif isinstance(model, (TMoEClipCA)):
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
                if isinstance(model, (TMoEClipCA, TMoEClipCA_C, TMoEClipCA_CO))
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

    if scheduler is not None:
        scheduler.step()
    return total_loss / len(dataloader)


def eval(
    model,
    mode,
    dataloader,
    criterion,
    alpha: float = 0.4,
    beta: float = None,
    theta: float = None,
    rec_weight: float = None,
    device: str = "cpu",
    loss_threshold: float = 0.2,
):
    model.eval()

    def _update_meter_set(meter_set, metrics):
        for k, v in metrics.items():
            meter_set.update(k, v)

    def _update_dataloader_metrics(tqdm_dataloader, meter_set):
        description_metrics = ["N%d" % k for k in [1, 5, 10, 20, 40]] + [
            "R%d" % k for k in [1, 5, 10, 20, 40]
        ]
        description = "Eval: " + ", ".join(s + " {:.4f}" for s in description_metrics)
        description = description.format(
            *(meter_set[k].avg for k in description_metrics)
        )
        tqdm_dataloader.set_description(description)

    total_loss = 0
    average_meter_set = AverageMeterSet()

    with torch.no_grad():
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
                    # if mode == "valid":
                    #     img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                    #     text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)
                    #     cosine_loss = (
                    #         1
                    #         - torch.cosine_similarity(
                    #             gen_mm, item_emb[labels], dim=-1
                    #         ).mean()
                    #     )
                    #     contra_loss = (
                    #         alpha * img_loss + beta * text_loss + theta * cosine_loss
                    #     )
                    #     wandb.log(
                    #         {
                    #             "valid_img_loss": img_loss.item(),
                    #             "valid_text_loss": text_loss.item(),
                    #             "valid_cosine_loss": cosine_loss.item(),
                    #             "valid_gate_mean": gate_mean.item(),
                    #         }
                    #     )
                    # rec_w = 1 - alpha - beta - theta
                elif isinstance(model, (TMoEClipCA_CO)):
                    gen_res, prompt_res, topk, item_emb, logits, gate_mean = model(
                        tokens, ori_emb, text_emb
                    )
                    # if mode == "valid":
                    #     img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                    #     text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)

                    #     topk = topk[:, -1, :].unsqueeze(-1)
                    #     mm_info = torch.stack((gen_emb, prompt_emb), dim=-2)
                    #     mm_info = torch.gather(
                    #         mm_info,
                    #         dim=2,
                    #         index=topk.expand(-1, -1, mm_info.size(-1)),
                    #     ).squeeze(-2)

                    #     cosine_loss = 1 - torch.cosine_similarity(
                    #         mm_info, item_emb[labels], dim=-1
                    #     )
                    #     mask = labels != 0
                    #     cosine_loss = cosine_loss[mask].mean()
                    #     contra_loss = (
                    #         alpha * img_loss + beta * text_loss + theta * cosine_loss
                    #     )

                    #     wandb.log(
                    #         {
                    #             "valid_img_loss": img_loss.item(),
                    #             "valid_text_loss": text_loss.item(),
                    #             "valid_cosine_loss": cosine_loss.item(),
                    #             "valid_gate_mean": gate_mean.item(),
                    #             "valid_a_gate_mean": torch.mean(
                    #                 topk.to(torch.float32)
                    #             ).item(),
                    #         }
                    #     )
                elif isinstance(model, (TMoEClipCA)):
                    gen_res, prompt_res, logits, gate_mean = model(
                        tokens, ori_emb, text_emb
                    )
                    # if mode == "valid":
                    #     img_loss = clip_loss(gen_res, gen_emb, model.logit_scale)
                    #     text_loss = clip_loss(prompt_res, prompt_emb, model.logit_scale)
                    #     contra_loss = alpha * img_loss + beta * text_loss
                    #     wandb.log(
                    #         {
                    #             "valid_img_loss": img_loss.item(),
                    #             "valid_text_loss": text_loss.item(),
                    #             "valid_gate_mean": gate_mean.item(),
                    #         }
                    #     )
                    # rec_w = 1 - alpha - beta
                elif isinstance(model, MoEClipCA):
                    enc_emb, logits = model(tokens, ori_emb, text_emb)
                    # contra_loss = (
                    #     alpha * clip_loss(enc_emb, gen_emb, model.logit_scale)
                    #     if mode == "valid"
                    #     else None
                    # )
                    # rec_w = 1 - alpha

                elif isinstance(model, CLIPCAModel):
                    enc_emb, logits = model(tokens, ori_emb)
                    # contra_loss = (
                    #     alpha * clip_loss(enc_emb, gen_emb, model.logit_scale)
                    #     if mode == "valid"
                    #     else None
                    # )
                    # rec_w = 1 - alpha

                elif isinstance(model, ARModel):
                    logits = model(tokens, ori_emb, gen_emb)
                elif isinstance(model, SASRec):
                    logits = model(tokens)

                logits = logits[:, -1, :]
                logits[:, 0] = logits.min(dim=-1).values - 1  # remove padding index

                labels = labels.squeeze()
                if mode == "valid":
                    rec_loss = criterion(logits, labels)
                    # if isinstance(model, CLIPCAModel):
                    #     loss = (
                    #         # rec_w * rec_loss + contra_loss
                    #         (rec_weight * rec_loss + contra_loss)
                    #         if alpha <= loss_threshold
                    #         else contra_loss
                    #     )
                    #     wandb.log(
                    #         {
                    #             "valid_contra_loss": contra_loss.item(),
                    #         }
                    #     )
                    # else:
                    #     loss = rec_loss

                    # t.set_postfix(loss=loss.item())
                    wandb.log(
                        {
                            # "valid_batch_loss": loss.item(),
                            "valid_rec_loss": rec_loss.item(),
                        }
                    )

                    # total_loss += loss

                metrics = modi_absolute_recall_mrr_ndcg_for_ks(logits, labels)
                _update_meter_set(average_meter_set, metrics)
                _update_dataloader_metrics(t, average_meter_set)
    average_metrics = average_meter_set.averages()

    # if mode == "valid":
    #     return total_loss / len(dataloader), average_metrics

    # pred_list = torch.cat(pred_list).tolist()
    return average_metrics
