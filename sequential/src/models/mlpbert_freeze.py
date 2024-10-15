from typing import Literal, Optional

import torch
from mlpbert import MLPBERT4Rec


class MLPwithBERTFreeze(MLPBERT4Rec):
    def __init__(
        self,
        num_item: int,
        num_cat: int,
        model_ckpt_path: str,
        gen_img_emb: torch.Tensor,  # TODO: try to remove
        item_prod_type: torch.Tensor,
        idx_groups: Optional[dict] = None,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 3,
        hidden_act: Literal["gelu", "mish", "silu"] = "gelu",
        num_gen_img: int = 1,
        max_len: int = 30,
        dropout_prob: float = 0.2,
        pos_emb: bool = True,
        cat_emb: bool = False,
        mlp_cat: bool = False,
        num_mlp_layers: int = 2,
        device: str = "cpu",
        text_emb: Optional[torch.Tensor] = None,
        merge: str = "concat",
        **kwargs,
    ):
        super(MLPwithBERTFreeze, self).__init__(
            num_item=num_item,
            num_cat=num_cat,
            gen_img_emb=gen_img_emb,
            item_prod_type=item_prod_type,
            idx_groups=idx_groups,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_act=hidden_act,
            num_gen_img=num_gen_img,
            max_len=max_len,
            dropout_prob=dropout_prob,
            pos_emb=pos_emb,
            cat_emb=cat_emb,
            mlp_cat=mlp_cat,
            num_mlp_layers=num_mlp_layers,
            device=device,
            text_emb=text_emb,
            merge=merge,
            **kwargs,
        )

        bert4rec_weight = torch.load(model_ckpt_path)

        try:
            incompat_keys = self.bert4rec_module.load_state_dict(
                bert4rec_weight, strict=False
            )
            assert incompat_keys.unexpected_keys == ["out.weight", "out.bias"]

            for param in self.bert4rec_module.parameters():
                param.requires_grad = False
        except AssertionError as e:
            print(f"Unexpected keys: {incompat_keys}")
            raise e
