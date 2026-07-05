import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def _scalar_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return value.detach().float().item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_trainable_loss(value):
    return torch.is_tensor(value) and value.numel() == 1 and value.requires_grad


def _is_loss_key(key):
    return key == "loss" or key.endswith("_loss")


def _should_track_scalar(key):
    return (
        "loss" in key
        or key.endswith("grad_norm")
        or key.startswith("pool_")
        or key.startswith("target_")
        or key.startswith("mixer/")
    )


def _grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.float().norm(2).item()
        total += param_norm ** 2
    return total ** 0.5


def _loss_grad_norm(loss, parameters):
    if not _is_trainable_loss(loss):
        return 0.0
    if not parameters:
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_norm = grad.detach().float().norm(2).item()
        total += grad_norm ** 2
    return total ** 0.5


def _iter_loss_grad_sources(ret):
    sources = {}
    explicit_sources = ret.get("_loss_grad_sources", {})
    if isinstance(explicit_sources, dict):
        for key, value in explicit_sources.items():
            if _is_loss_key(key):
                sources[key] = value

    for key, value in ret.items():
        if key == "_loss_grad_sources":
            continue
        if _is_loss_key(key) and key not in sources:
            sources[key] = value
    return sources.items()


def _update_meter(meters, key, value, batch_size):
    if key not in meters:
        meters[key] = AverageMeter()
    meters[key].update(value, batch_size)


def _set_loader_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    batch_sampler_inner = getattr(batch_sampler, "sampler", None)
    if hasattr(batch_sampler_inner, "set_epoch") and batch_sampler_inner is not sampler:
        batch_sampler_inner.set_epoch(epoch)


def _move_train_batch_to_device(batch, device, pnp_text_only=False):
    if pnp_text_only:
        return {
            key: value.to(device)
            for key, value in batch.items()
            if key != "images"
        }
    return {key: value.to(device) for key, value in batch.items()}


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _count_module_tensors(module, name_filter=None):
    def _include(name):
        return name_filter is None or name_filter(name)

    total_params = 0
    trainable_params = 0
    for name, parameter in module.named_parameters():
        if not _include(name):
            continue
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    buffer_params = sum(
        buffer.numel()
        for name, buffer in module.named_buffers()
        if _include(name)
    )
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "buffers": buffer_params,
    }


def _log_param_scope(logger, label, stats):
    logger.info(
        "%s params: total=%s (%.3fM), trainable=%s (%.3fM), frozen=%s (%.3fM), "
        "buffers=%s (%.3fM)",
        label,
        f"{stats['total_params']:,}",
        stats["total_params"] / 1_000_000.0,
        f"{stats['trainable_params']:,}",
        stats["trainable_params"] / 1_000_000.0,
        f"{stats['frozen_params']:,}",
        stats["frozen_params"] / 1_000_000.0,
        f"{stats['buffers']:,}",
        stats["buffers"] / 1_000_000.0,
    )


def _log_enrichment_branch_size(model, logger):
    model = _unwrap_model(model)
    target_enricher = getattr(model, "target_enricher", None)
    model_stats = _count_module_tensors(model)
    host_stats = _count_module_tensors(
        model,
        name_filter=lambda name: not name.startswith("target_enricher."),
    )
    _log_param_scope(logger, "Model", model_stats)
    _log_param_scope(logger, "Host", host_stats)

    if target_enricher is None:
        logger.info("Target enrichment branch params: disabled")
        return
    _log_param_scope(
        logger,
        "Target enrichment branch",
        _count_module_tensors(target_enricher),
    )


def _target_enrichment_active(args, epoch):
    enrichment_start = getattr(args, "enrichment_start", 1)
    if enrichment_start < 1:
        raise ValueError("--enrichment_start must be a positive integer")
    return getattr(args, "target_enrichment", False) and epoch >= enrichment_start


def _should_run_eval(args, epoch):
    eval_after_epoch = getattr(args, "eval_after_epoch", 0)
    if eval_after_epoch < 0:
        raise ValueError("--eval_after_epoch must be a non-negative integer")
    return epoch >= eval_after_epoch


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, target_pool=None):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")
    logger.info(f"training loader contains {len(train_loader)} batches per epoch")
    if get_rank() == 0:
        _log_enrichment_branch_size(model, logger)
    if target_pool is not None and getattr(args, "enrichment_start", 1) > 1:
        logger.info(
            "Target enrichment delayed until epoch {}; earlier epochs use host training only".format(
                args.enrichment_start
            )
        )
    if getattr(args, "eval_after_epoch", 0) > 0:
        logger.info(
            "Evaluation delayed until epoch {}; earlier epochs skip validation".format(
                args.eval_after_epoch
            )
        )

    meters = {
        "loss": AverageMeter(),
        "host_loss": AverageMeter(),
        "target_enrichment_loss": AverageMeter(),
        "target_retrieval_loss": AverageMeter(),
        "grad_norm": AverageMeter(),
        "loss_grad_norm": AverageMeter(),
        "host_loss_grad_norm": AverageMeter(),
        "target_enrichment_loss_grad_norm": AverageMeter(),
        "target_retrieval_loss_grad_norm": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0
    best_epoch = None
    current_steps = 0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()
        _set_loader_epoch(train_loader, epoch)
        use_target_enrichment = target_pool is not None and _target_enrichment_active(args, epoch)
        if target_pool is not None and epoch == getattr(args, "enrichment_start", 1):
            logger.info("Target enrichment starts at epoch {}".format(epoch))
        logger.info(f"Epoch[{epoch}/{num_epoch}] started")

        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = _move_train_batch_to_device(
                batch,
                device,
                pnp_text_only=getattr(args, "pnp_text_only", False),
            )

            target_cache = None
            if use_target_enrichment:
                target_cache = target_pool.get_train_cache(model, batch, epoch, current_steps)

            ret = model(
                batch,
                epoch=epoch,
                current_step=current_steps,
                target_cache=target_cache,
            )
            if target_cache is not None and "diagnostics" in target_cache:
                for diag_key, diag_value in target_cache["diagnostics"].items():
                    if isinstance(diag_value, (int, float)):
                        ret[diag_key] = diag_value

            total_loss = ret.get("loss")
            if total_loss is None:
                total_loss = sum(
                    value
                    for key, value in ret.items()
                    if "loss" in key and _is_trainable_loss(value)
                )

            batch_size = batch["caption_ids"].shape[0]
            loss_scalar = _scalar_value(total_loss)
            if loss_scalar is None:
                raise ValueError("Training forward must return a scalar loss")
            meters["loss"].update(loss_scalar, batch_size)

            for key, value in ret.items():
                if key == "loss":
                    continue
                scalar = _scalar_value(value)
                if scalar is None or not _should_track_scalar(key):
                    continue
                _update_meter(meters, key, scalar, batch_size)

            optimizer.zero_grad()
            trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            loss_grad_sources = dict(_iter_loss_grad_sources(ret))
            loss_grad_sources["loss"] = total_loss
            for loss_key, loss_value in loss_grad_sources.items():
                grad_norm_key = f"{loss_key}_grad_norm"
                grad_norm_value = _loss_grad_norm(loss_value, trainable_params)
                _update_meter(meters, grad_norm_key, grad_norm_value, batch_size)

            total_loss.backward()
            grad_norm_value = _grad_norm(model.parameters())
            meters["grad_norm"].update(grad_norm_value, batch_size)
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for key, meter in meters.items():
                    if meter.count > 0:
                        info_str += f", {key}: {meter.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        temperature = _scalar_value(ret.get("temperature"))
        if temperature is not None:
            tb_writer.add_scalar("temperature", temperature, epoch)
        for key, meter in meters.items():
            if meter.count > 0:
                tb_writer.add_scalar(key, meter.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / time_per_batch,
                )
            )

        if epoch % eval_period == 0 and _should_run_eval(args, epoch):
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                eval_model = model.module.eval() if args.distributed else model.eval()
                top1 = evaluator.eval(
                    eval_model,
                    use_target_enrichment=_target_enrichment_active(args, epoch),
                )
                torch.cuda.empty_cache()
                top1 = float(top1)
                if best_top1 < top1:
                    best_top1 = top1
                    best_epoch = epoch
                    arguments["epoch"] = epoch
                    arguments["best_top1"] = best_top1
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        if best_epoch is None:
            logger.info("No validation results were produced; best checkpoint was not saved.")
        else:
            logger.info(f"best R1: {best_top1} at epoch {best_epoch}")


def do_inference(model, test_img_loader, test_txt_loader, args=None):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    return evaluator.eval(
        model.eval(),
        use_target_enrichment=getattr(args, "target_enrichment", False) if args is not None else False,
    )
