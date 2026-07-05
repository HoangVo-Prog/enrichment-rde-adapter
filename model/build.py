from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from .enrichment import (
    TargetPrototypeEnricher,
    build_evidence_bank,
    evidence_slot_indices,
    finalize_target_evidence_cache,
)


def freeze_host_parameters(model, trainable_prefix="target_enricher."):
    frozen_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        keep_trainable = name.startswith(trainable_prefix)
        parameter.requires_grad = keep_trainable
        if keep_trainable:
            trainable_params += parameter.numel()
        else:
            frozen_params += parameter.numel()
    if trainable_params == 0:
        raise ValueError("--freeze_host requires target enrichment parameters to train")
    return frozen_params, trainable_params


def _set_default_attr(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _ensure_target_enrichment_defaults(args):
    _set_default_attr(args, "only_global", True)
    _set_default_attr(args, "target_enrichment", False)
    _set_default_attr(args, "enrichment_start", 1)
    _set_default_attr(args, "enrichment_space", "global")
    _set_default_attr(args, "top_m", 32)
    _set_default_attr(args, "topm_rank_space", "host_global")
    _set_default_attr(args, "topm_rank_lambda", 0.5)
    _set_default_attr(args, "lambda_ret", 1.0)
    _set_default_attr(args, "tau", 0.015)
    _set_default_attr(args, "freeze_host", False)
    _set_default_attr(args, "finetune_clip", False)
    _set_default_attr(args, "use_host_loss", True)
    _set_default_attr(args, "lambda_host", 1.0)
    _set_default_attr(args, "extractor_mode", "global,horizontal")
    _set_default_attr(args, "num_parts", 6)
    _set_default_attr(args, "target_relative_space", "host_global")
    _set_default_attr(args, "target_relative_num_clusters", 16)
    _set_default_attr(args, "target_relative_cluster_method", "kmeans")
    _set_default_attr(args, "evidence_projection", "auto")
    _set_default_attr(args, "context_module", "mixer")
    _set_default_attr(args, "mixer_dim", 256)
    _set_default_attr(args, "mixer_depth", 2)
    _set_default_attr(args, "mixer_hidden_part", 32)
    _set_default_attr(args, "mixer_hidden_rank", 64)
    _set_default_attr(args, "mixer_hidden_channel", 512)
    _set_default_attr(args, "mixer_hidden_readout", 128)
    _set_default_attr(args, "context_pooling", "mlp")
    _set_default_attr(args, "residual_gate", "residual")
    _set_default_attr(args, "enrich_gamma", None)
    _set_default_attr(args, "residual_gate_hidden_dim", 128)
    
class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        _ensure_target_enrichment_defaults(args)
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg, state_dict = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, args.num_experts, args.topk, args.reduction)

        self.embed_dim = base_cfg['embed_dim']
        self.grab_embed_dim = self.embed_dim
        if getattr(args, "target_enrichment", False):
            if getattr(args, "enrichment_space", "global") == "grab":
                raise ValueError(
                    "--enrichment_space grab requires alternate/GRAB features, "
                    "which this host does not expose"
                )
            if getattr(args, "topm_rank_space", "host_global") == "hybrid_global_grab":
                raise ValueError(
                    "--topm_rank_space hybrid_global_grab requires alternate/GRAB features, "
                    "which this host does not expose"
                )

        # new add vs V5
        self.apply(self.init_weights) # random init must before loading pretrain
        self.base_model.load_param(state_dict)

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
            
        if 'id' in args.loss_names or 'imkt' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        
        for i in range(12):
            for j in range(args.num_experts):
                nn.init.kaiming_uniform_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].down.bias)
                nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].up.weight)
                nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].feed_forward.experts[j].up.bias)


        for i in range(12):
            for j in range(args.num_experts):
                nn.init.kaiming_uniform_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].down.bias)
                nn.init.zeros_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].up.weight)
                nn.init.zeros_(self.base_model.transformer.resblocks[i].feed_forward.experts[j].up.bias)

        if getattr(args, "target_enrichment", False):
            self.target_enricher = TargetPrototypeEnricher(
                self.embed_dim,
                self.grab_embed_dim,
                args,
            )

        self.freeze_host_stats = None
        if getattr(args, "freeze_host", False):
            self.freeze_host_stats = freeze_host_parameters(self)
                
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()              

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image, l_aux=0):
        outputs = self.base_model.encode_image(image, l_aux)
        x = outputs[0]
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text, l_aux=0):
        outputs = self.base_model.encode_text(text, l_aux)
        x = outputs[0]
        return x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)].float()

    def encode_eval_text_bundle(self, text, include_grab=False):
        if include_grab:
            raise ValueError("This host does not expose alternate/GRAB text features")
        return {"host_features": self.encode_text(text, l_aux=0)}

    def encode_eval_image_bundle(self, image, include_grab=False, cache_target=False, cache_prototypes=True):
        if include_grab:
            raise ValueError("This host does not expose alternate/GRAB image features")

        image_feats, _ = self.base_model.encode_image(image, l_aux=0)
        host_features = image_feats[:, 0, :].float()
        bundle = {"host_features": host_features}

        if not cache_target:
            return bundle

        cache = {"host_image_features": host_features}
        if not cache_prototypes:
            bundle["target_cache"] = cache
            return bundle

        if getattr(self.args, "enrichment_space", "global") == "grab":
            raise ValueError("This host does not expose alternate/GRAB retrieval features")
        if getattr(self.args, "topm_rank_space", "host_global") == "hybrid_global_grab":
            raise ValueError("This host does not expose alternate/GRAB ranking features")

        cache["retrieval_features"] = host_features

        grid_size = None
        if hasattr(self.base_model.visual, "num_y") and hasattr(self.base_model.visual, "num_x"):
            grid_size = (self.base_model.visual.num_y, self.base_model.visual.num_x)
        evidence_bank = build_evidence_bank(
            image_feats,
            getattr(self.args, "num_parts", 6),
            grid_size=grid_size,
            mode=getattr(self.args, "extractor_mode", "global,horizontal"),
            retrieval_features=cache["retrieval_features"],
        )
        cache["evidence_bank"] = evidence_bank
        cache["prototypes"] = evidence_bank

        slots = evidence_slot_indices(
            getattr(self.args, "extractor_mode", "global,horizontal"),
            getattr(self.args, "num_parts", 6),
        )
        if (
            "retrieval_backbone" in slots
            and cache["retrieval_features"].shape[-1] != self.embed_dim
        ):
            if getattr(self.args, "evidence_projection", "auto") == "none":
                raise ValueError(
                    "--extractor_mode retrieval_backbone requires evidence projection "
                    "when retrieval feature dim differs from the shared evidence dim"
                )
            cache["retrieval_backbone_features"] = cache["retrieval_features"]

        bundle["target_cache"] = cache
        return bundle

    def encode_target_image_cache(self, image, cache_prototypes=True):
        return self.encode_eval_image_bundle(
            image,
            include_grab=False,
            cache_target=True,
            cache_prototypes=cache_prototypes,
        )["target_cache"]

    def finalize_target_cache(self, cache):
        return finalize_target_evidence_cache(cache, self.args, self.embed_dim)

    def enrich_text_features(self, query_features, host_text_features, target_cache, grab_text_features=None):
        if grab_text_features is not None:
            raise ValueError("This host does not expose alternate/GRAB text features")
        self.target_enricher = self.target_enricher.float()
        return self.target_enricher.enrich_only(
            query_features=query_features,
            host_text_features=host_text_features,
            pool_cache=target_cache,
            space=getattr(self.args, "enrichment_space", "global"),
            grab_text_features=None,
        )

    def forward(self, batch, epoch=None, current_step=None, target_cache=None):
        ret = dict()
        use_host_loss = getattr(self.args, "use_host_loss", True)
        pnp_text_only = getattr(self.args, "pnp_text_only", False)

        caption_ids = batch['caption_ids']
        if pnp_text_only:
            text_feats, _ = self.base_model.encode_text(caption_ids.long(), l_aux=0)
            t_feats = text_feats[
                torch.arange(text_feats.shape[0], device=text_feats.device),
                caption_ids.argmax(dim=-1),
            ].float()
            image_feats = None
            i_feats = None
            l_aux = t_feats.float().sum() * 0.0
        else:
            images = batch['images']
            image_feats, text_feats, l_aux = self.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            # i_feats = image_feats.float() # for CLIP ResNet visual model

            # todo
            t_feats = text_feats[
                torch.arange(text_feats.shape[0], device=text_feats.device),
                caption_ids.argmax(dim=-1),
            ].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if getattr(self.args, "target_enrichment", False) and target_cache is not None:
            self.target_enricher = self.target_enricher.float()
            target_ret = self.target_enricher(
                query_features=t_feats,
                host_text_features=t_feats,
                grab_text_features=None,
                query_pids=batch["pids"],
                pool_cache=target_cache,
                space="global",
            )
            t_feats = target_ret["enriched_features"]
            target_metrics = {
                "target_enrichment_loss": target_ret["total_loss"],
                "target_retrieval_loss": target_ret["target_retrieval_loss"].detach(),
                "_loss_grad_sources": {
                    "target_enrichment_loss": target_ret["total_loss"],
                    "target_retrieval_loss": target_ret["target_retrieval_loss"],
                },
            }
            for metric_key, metric_value in target_ret.items():
                if not (metric_key.startswith("target_") or metric_key.startswith("mixer/")):
                    continue
                if torch.is_tensor(metric_value) and metric_value.numel() == 1:
                    target_metrics[metric_key] = metric_value.detach()
            ret.update(target_metrics)

        if use_host_loss and 'aux' in self.current_task:
            #print(f'l_aux:{l_aux}')
            ret.update({'aux_loss': 0.5 * l_aux})

        if use_host_loss and 'triplet_enhance' in self.current_task:
            ret.update({'triplet_enhance_loss': 0.5 * objectives.compute_triplet_enhance(i_feats, t_feats, batch['pids'])})

        if use_host_loss and 'triplet_enhance_shuffle' in self.current_task:
            ret.update({'triplet_enhance_shuffle_loss': 0.5 * objectives.compute_triplet_enhance_shuffle(i_feats, t_feats, batch['pids'])})

        if use_host_loss and 'triplet' in self.current_task:
            ret.update({'triplet_loss':0.5 * objectives.compute_triplet(i_feats, t_feats)})

        if use_host_loss and 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if use_host_loss and 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if use_host_loss and 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if use_host_loss and 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if use_host_loss and 'imkt' in self.current_task:
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'imkt_loss': objectives.compute_imkt(text_logits, batch['pids'])})
        
        if use_host_loss and 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids, l_aux=0)[0]

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)

            mlm_labels = batch['mlm_labels'].reshape(-1)


            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        zero_source = t_feats if pnp_text_only else i_feats
        zero = zero_source.float().sum() * 0.0
        host_loss_terms = [
            value
            for key, value in ret.items()
            if key.endswith("_loss")
            and key not in ("host_loss", "target_enrichment_loss", "target_retrieval_loss")
            and torch.is_tensor(value)
        ]
        if use_host_loss and host_loss_terms:
            host_loss = getattr(self.args, "lambda_host", 1.0) * sum(host_loss_terms)
        else:
            host_loss = zero
        target_enrichment_loss = ret.get("target_enrichment_loss", zero)
        ret.update({
            "host_loss": host_loss,
            "loss": host_loss + target_enrichment_loss,
        })

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
