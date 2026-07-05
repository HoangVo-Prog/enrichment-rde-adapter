import logging
import re
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    last_rel_rank = (tmp_cmc != num_rel.view(-1, 1)).sum(1) + 1
    mINP = (num_rel.float() / last_rel_rank.float()).mean() * 100

    rank_positions = torch.arange(
        1,
        tmp_cmc.shape[1] + 1,
        device=tmp_cmc.device,
        dtype=torch.float32,
    ).view(1, -1)
    tmp_cmc = tmp_cmc.float()
    tmp_cmc.div_(rank_positions)
    tmp_cmc.mul_(matches)
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, name, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
    )
    t2i_cmc = t2i_cmc.numpy()
    t2i_mAP = t2i_mAP.numpy()
    t2i_mINP = t2i_mINP.numpy()
    row = [
        name,
        t2i_cmc[0],
        t2i_cmc[4],
        t2i_cmc[9],
        t2i_mAP,
        t2i_mINP,
        t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9],
    ]
    if retur_indices:
        return row, indices
    return row


def _metric_task_name(task):
    task = str(task).replace("+", "_plus_")
    task = task.replace("(", "_").replace(")", "")
    task = task.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_/-]+", "_", task).strip("_")


def _row_to_eval_metrics(row):
    task = _metric_task_name(row[0])
    return {
        f"eval/{task}/R1": float(row[1]),
        f"eval/{task}/R5": float(row[2]),
        f"eval/{task}/R10": float(row[3]),
        f"eval/{task}/mAP": float(row[4]),
        f"eval/{task}/mINP": float(row[5]),
        f"eval/{task}/rSum": float(row[6]) if len(row) > 6 else 0.0,
    }


def _rename_metric_row(row, task_name):
    return [task_name, *row[1:]]


def _scale_scores_like(scores, reference, eps=1e-12):
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    ref_min = reference.min(dim=1, keepdim=True).values
    ref_max = reference.max(dim=1, keepdim=True).values

    score_range = (score_max - score_min).clamp_min(eps)
    ref_range = ref_max - ref_min
    return (scores - score_min) / score_range * ref_range + ref_min


def _prototype_lambdas():
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _format_lambda(value):
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return "{:.2f}".format(value).rstrip("0").rstrip(".")


def _ablation_lambda_from_key(key):
    match = re.search(r"\(([-+]?\d*\.?\d+)\)$", str(key))
    return float(match.group(1)) if match else 0.0


def _default_args():
    return SimpleNamespace(
        only_global=True,
        target_enrichment=False,
        enrichment_space="global",
        topm_rank_space="host_global",
    )


class Evaluator:
    def __init__(self, img_loader, txt_loader, args=None):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.args = args if args is not None else _default_args()
        if not hasattr(self.args, "only_global"):
            self.args.only_global = True
        if not hasattr(self.args, "target_enrichment"):
            self.args.target_enrichment = False
        if not hasattr(self.args, "enrichment_space"):
            self.args.enrichment_space = "global"
        if not hasattr(self.args, "topm_rank_space"):
            self.args.topm_rank_space = "host_global"
        self.logger = logging.getLogger("IRRA.eval")
        self.last_metrics = {}
        self.last_best_task = None

    def _core_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _cache_chunk_to_cpu(self, cache):
        return {key: value.detach().cpu() for key, value in cache.items()}

    def _finalize_target_cache(self, model, gids, cache_chunks, device):
        target_cache = {}
        for key in cache_chunks[0].keys():
            target_cache[key] = torch.cat(
                [chunk[key] for chunk in cache_chunks],
                dim=0,
            ).to(device)
        target_cache["pids"] = gids.to(device)
        core_model = self._core_model(model)
        if hasattr(core_model, "finalize_target_cache"):
            target_cache = core_model.finalize_target_cache(target_cache)
        return target_cache

    def _compute_text_branches(self, model, include_grab=False):
        model = model.eval()
        device = next(model.parameters()).device
        core_model = self._core_model(model)

        qids, host_feats, grab_feats, batch_sizes = [], [], [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                if hasattr(core_model, "encode_eval_text_bundle"):
                    bundle = core_model.encode_eval_text_bundle(
                        caption,
                        include_grab=include_grab,
                    )
                    host_feat = bundle["host_features"].cpu()
                    grab_feat = bundle["grab_features"].cpu() if include_grab else None
                else:
                    host_feat = core_model.encode_text(caption).cpu()
                    grab_feat = core_model.encode_text_grab(caption).cpu() if include_grab else None
            flat_pid = pid.view(-1)
            qids.append(flat_pid)
            host_feats.append(host_feat)
            if include_grab:
                grab_feats.append(grab_feat)
            batch_sizes.append(int(flat_pid.numel()))

        host_feats = torch.cat(host_feats, 0).cpu()
        qids = torch.cat(qids, 0).cpu()
        grab_feats = torch.cat(grab_feats, 0).cpu() if include_grab else None
        return host_feats, grab_feats, qids, batch_sizes

    def _compute_image_branches(
        self,
        model,
        include_grab=False,
        include_target_cache=False,
    ):
        model = model.eval()
        device = next(model.parameters()).device
        core_model = self._core_model(model)

        gids, host_feats, grab_feats, cache_chunks = [], [], [], []
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                if hasattr(core_model, "encode_eval_image_bundle"):
                    bundle = core_model.encode_eval_image_bundle(
                        img,
                        include_grab=include_grab,
                        cache_target=include_target_cache,
                    )
                    host_feat = bundle["host_features"].cpu()
                    grab_feat = bundle["grab_features"].cpu() if include_grab else None
                    cache = bundle.get("target_cache")
                else:
                    host_feat = core_model.encode_image(img).cpu()
                    grab_feat = core_model.encode_image_grab(img).cpu() if include_grab else None
                    cache = core_model.encode_target_image_cache(img) if include_target_cache else None
            gids.append(pid.view(-1))
            host_feats.append(host_feat)
            if include_grab:
                grab_feats.append(grab_feat)
            if include_target_cache:
                cache_chunks.append(self._cache_chunk_to_cpu(cache))

        gids = torch.cat(gids, 0).cpu()
        host_feats = torch.cat(host_feats, 0).cpu()
        grab_feats = torch.cat(grab_feats, 0).cpu() if include_grab else None

        target_cache = None
        if include_target_cache:
            target_cache = self._finalize_target_cache(
                model,
                gids,
                cache_chunks,
                device,
            )
        return host_feats, grab_feats, gids, target_cache

    def _compute_embedding(self, model):
        qfeats, _, qids, _ = self._compute_text_branches(model, include_grab=False)
        gfeats, _, gids, _ = self._compute_image_branches(model, include_grab=False)
        return qfeats, gfeats, qids, gids

    def _run_enrich_text_features(
        self,
        model,
        target_cache,
        host_text_feat,
        grab_text_feat,
    ):
        query_feat = grab_text_feat if self.args.enrichment_space == "grab" else host_text_feat
        core_model = self._core_model(model)
        with torch.no_grad():
            return core_model.enrich_text_features(
                query_feat,
                host_text_feat,
                target_cache,
                grab_text_features=grab_text_feat,
            ).cpu()

    def _enrich_text_features_adaptive(
        self,
        model,
        target_cache,
        host_text_feat,
        grab_text_feat,
    ):
        try:
            return self._run_enrich_text_features(
                model,
                target_cache,
                host_text_feat,
                grab_text_feat,
            )
        except torch.cuda.OutOfMemoryError as error:
            batch_size = host_text_feat.shape[0]
            if batch_size <= 1:
                raise
            error.__traceback__ = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            split = batch_size // 2
            self.logger.warning(
                "CUDA OOM during enriched text evaluation for batch size {}; "
                "retrying as {} + {}".format(batch_size, split, batch_size - split)
            )
            left_grab = grab_text_feat[:split] if grab_text_feat is not None else None
            right_grab = grab_text_feat[split:] if grab_text_feat is not None else None
            left = self._enrich_text_features_adaptive(
                model,
                target_cache,
                host_text_feat[:split],
                left_grab,
            )
            right = self._enrich_text_features_adaptive(
                model,
                target_cache,
                host_text_feat[split:],
                right_grab,
            )
            return torch.cat([left, right], dim=0)

    def _compute_enriched_text_embedding_from_features(
        self,
        model,
        target_cache,
        host_text_features,
        qids,
        batch_sizes,
        grab_text_features=None,
    ):
        model = model.eval()
        device = next(model.parameters()).device

        needs_grab = (
            self.args.enrichment_space == "grab"
            or getattr(self.args, "topm_rank_space", "host_global") == "hybrid_global_grab"
        )
        qfeats = []
        offset = 0
        for batch_size in batch_sizes:
            end = offset + batch_size
            host_text_feat = host_text_features[offset:end].to(device)
            grab_text_feat = None
            if needs_grab:
                if grab_text_features is None:
                    raise ValueError("Target enrichment requires cached GRAB text features")
                grab_text_feat = grab_text_features[offset:end].to(device)
            text_feat = self._enrich_text_features_adaptive(
                model,
                target_cache,
                host_text_feat,
                grab_text_feat,
            )
            qfeats.append(text_feat)
            offset = end

        return torch.cat(qfeats, 0).cpu(), qids.cpu()

    def _build_base_tasks(self, sims_global, sims_grab):
        return [("global", sims_global)]

    def _iter_eval_tasks(self, base_tasks, sims_target):
        for task_name, task_scores in base_tasks:
            yield task_name, task_scores

        if sims_target is None:
            return

        scaled_base_tasks = [
            (base_name, _scale_scores_like(base_scores, sims_target))
            for base_name, base_scores in base_tasks
        ]
        for proto_lambda in _prototype_lambdas():
            proto_value = _format_lambda(proto_lambda)
            for base_name, scaled_base_scores in scaled_base_tasks:
                fused_name = "{}+proto({})".format(base_name, proto_value)
                if abs(proto_lambda - 1.0) < 1e-12:
                    yield fused_name, sims_target
                else:
                    yield fused_name, (
                        (1.0 - proto_lambda) * scaled_base_scores
                        + proto_lambda * sims_target
                    )

    def eval(self, model, i2t_metric=False, use_target_enrichment=None):
        if use_target_enrichment is None:
            use_target_enrichment = getattr(self.args, "target_enrichment", False)

        include_grab = False
        host_qfeats, grab_qfeats, qids, text_batch_sizes = self._compute_text_branches(
            model,
            include_grab=include_grab,
        )
        host_gfeats, grab_gfeats, gids, target_cache = self._compute_image_branches(
            model,
            include_grab=include_grab,
            include_target_cache=use_target_enrichment,
        )
        qfeats = F.normalize(host_qfeats, p=2, dim=1)
        gfeats = F.normalize(host_gfeats, p=2, dim=1)
        sims_global = qfeats @ gfeats.t()

        base_tasks = self._build_base_tasks(sims_global, None)
        sims_target = None
        if use_target_enrichment:
            target_qfeats, target_qids = self._compute_enriched_text_embedding_from_features(
                model,
                target_cache,
                host_qfeats,
                qids,
                text_batch_sizes,
                grab_text_features=grab_qfeats,
            )
            target_qfeats = F.normalize(target_qfeats, p=2, dim=1)
            target_gfeats = F.normalize(
                target_cache["retrieval_features"].detach().cpu(),
                p=2,
                dim=1,
            )
            sims_target = target_qfeats @ target_gfeats.t()
            qids = target_qids

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])

        top1 = 0
        eval_metrics = {}
        rows_by_task = {}
        best_task = None
        best_row = None
        best_ablation_task = None
        best_ablation_row = None
        t2i_metric_cache = {}
        i2t_metric_cache = {}

        for key, sims in self._iter_eval_tasks(base_tasks, sims_target):
            t2i_name = f"{key}-t2i"
            cache_key = id(sims) if sims_target is not None and sims is sims_target else None
            if cache_key is not None and cache_key in t2i_metric_cache:
                row = _rename_metric_row(t2i_metric_cache[cache_key], t2i_name)
            else:
                row = get_metrics(sims, qids, gids, t2i_name, False)
                if cache_key is not None:
                    t2i_metric_cache[cache_key] = row
            table.add_row(row)
            rows_by_task[key] = row
            eval_metrics.update(_row_to_eval_metrics(row))

            if i2t_metric:
                i2t_name = f"{key}-i2t"
                if cache_key is not None and cache_key in i2t_metric_cache:
                    i2t_row = _rename_metric_row(i2t_metric_cache[cache_key], i2t_name)
                else:
                    i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                        similarity=sims.t(),
                        q_pids=gids,
                        g_pids=qids,
                        max_rank=10,
                        get_mAP=True,
                    )
                    i2t_cmc = i2t_cmc.numpy()
                    i2t_mAP = i2t_mAP.numpy()
                    i2t_mINP = i2t_mINP.numpy()
                    i2t_row = [
                        i2t_name,
                        i2t_cmc[0],
                        i2t_cmc[4],
                        i2t_cmc[9],
                        i2t_mAP,
                        i2t_mINP,
                        i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9],
                    ]
                    if cache_key is not None:
                        i2t_metric_cache[cache_key] = i2t_row
                table.add_row(i2t_row)
                eval_metrics.update(_row_to_eval_metrics(i2t_row))

            if best_row is None or row[1] > best_row[1]:
                best_task = key
                best_row = row
            if "+proto(" in key and (best_ablation_row is None or row[1] > best_ablation_row[1]):
                best_ablation_task = key
                best_ablation_row = row

        if best_ablation_row is not None:
            top1 = float(best_ablation_row[1])
            best_task = best_ablation_task
            eval_metrics["eval/ablation_best_R1"] = float(best_ablation_row[1])
            eval_metrics["eval/ablation_best_R5"] = float(best_ablation_row[2])
            eval_metrics["eval/ablation_best_R10"] = float(best_ablation_row[3])
            eval_metrics["eval/ablation_best_mAP"] = float(best_ablation_row[4])
            eval_metrics["eval/ablation_best_mINP"] = float(best_ablation_row[5])
            eval_metrics["eval/ablation_best_rSum"] = float(best_ablation_row[6])
            eval_metrics["eval/ablation_best_lambda"] = _ablation_lambda_from_key(best_ablation_task)
        elif best_row is not None:
            top1 = float(best_row[1])

        target_key = "global+proto(1)"
        if "global" in rows_by_task and target_key in rows_by_task:
            global_row = rows_by_task["global"]
            target_row = rows_by_task[target_key]
            eval_metrics["eval/delta_R1_target_vs_global"] = float(target_row[1] - global_row[1])
            eval_metrics["eval/delta_R5_target_vs_global"] = float(target_row[2] - global_row[2])
            eval_metrics["eval/delta_R10_target_vs_global"] = float(target_row[3] - global_row[3])
            eval_metrics["eval/delta_mAP_target_vs_global"] = float(target_row[4] - global_row[4])
            eval_metrics["eval/delta_mINP_target_vs_global"] = float(target_row[5] - global_row[5])
            eval_metrics["eval/delta_rSum_target_vs_global"] = float(target_row[6] - global_row[6])

        self.last_metrics = eval_metrics
        self.last_best_task = best_task

        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["rSum"] = lambda f, v: f"{v:.3f}"
        self.logger.info("\n" + str(table))
        self.logger.info("\n" + "best R1 = " + str(top1))
        if best_task is not None:
            self.logger.info("best R1 row = {}".format(best_task))

        return top1
