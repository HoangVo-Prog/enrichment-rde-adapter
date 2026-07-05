import os
import os.path as op
import torch
import time
import warnings


from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer, delete_output_checkpoints
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from model.enrichment import TargetPoolManager
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize
from utils.reproducibility import configure_reproducibility
from utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    upload_best_checkpoint_artifact,
    upload_checkpoint_artifacts,
)

warnings.filterwarnings("ignore")


_RDE_ADAPTER_TRAINABLE_NAMES = (
    "adapter_mlp",
    "ln_3",
    "experts",
    "feed_forward",
    "ln_4",
    "param",
    "v2i_proj",
    "task_param",
)


def _strip_module_prefix(key):
    return key[7:] if key.startswith("module.") else key


def _unwrap_checkpoint_state_dict(checkpoint):
    for key in ("model", "state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint must be a state dict or contain a 'model'/'state_dict' entry")


def _iter_finetune_candidates(raw_key, base_model_subkeys):
    key = _strip_module_prefix(raw_key)
    direct_candidates = [key]
    if key.startswith("model."):
        direct_candidates.append(key[len("model."):])

    candidates = []
    seen = set()
    for candidate in direct_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
        if not candidate.startswith("base_model.") and candidate in base_model_subkeys:
            mapped = "base_model." + candidate
            if mapped not in seen:
                seen.add(mapped)
                candidates.append(mapped)
    return candidates


def _finetune_clip_checkpoint(args):
    value = getattr(args, "finetune_clip", False)
    return value if isinstance(value, str) and value else ""


def _finetune_clip_enabled(args):
    return bool(getattr(args, "finetune_clip", False))


def load_finetune_clip_checkpoint(model, checkpoint_file, logger):
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
    except RuntimeError as torch_load_error:
        try:
            checkpoint = torch.jit.load(checkpoint_file, map_location='cpu').state_dict()
        except RuntimeError:
            raise torch_load_error

    state_dict = _unwrap_checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    base_model_subkeys = {
        key[len("base_model."):]
        for key in model_state.keys()
        if key.startswith("base_model.")
    }
    update_state = {}
    skipped_target = 0
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_tensor = 0

    for raw_key, value in state_dict.items():
        normalized_key = _strip_module_prefix(raw_key)
        if normalized_key.startswith("target_enricher.") or normalized_key.startswith("model.target_enricher."):
            skipped_target += 1
            continue
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue

        has_name_match = False
        loaded = False
        for candidate_key in _iter_finetune_candidates(raw_key, base_model_subkeys):
            if candidate_key not in model_state:
                continue
            has_name_match = True
            if model_state[candidate_key].shape != value.shape:
                continue
            update_state[candidate_key] = value.detach().clone()
            loaded = True
            break

        if loaded:
            continue
        if has_name_match:
            skipped_shape += 1
        else:
            skipped_missing += 1

    if not update_state:
        raise RuntimeError(f"No compatible weights found in --finetune_clip checkpoint: {checkpoint_file}")

    model_state.update(update_state)
    model.load_state_dict(model_state)
    logger.info(
        "Loaded %d tensors from --finetune_clip checkpoint %s; skipped %d target-enrichment, %d missing, %d shape-mismatch, %d non-tensor entries",
        len(update_state),
        checkpoint_file,
        skipped_target,
        skipped_missing,
        skipped_shape,
        skipped_non_tensor,
    )


def _count_parameters(model, name_filter=None):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        normalized_name = _strip_module_prefix(name)
        if name_filter is not None and not name_filter(normalized_name):
            continue
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable


def _is_base_model_parameter(name):
    return _strip_module_prefix(name).startswith("base_model.")


def _is_rde_adapter_parameter(name):
    normalized_name = _strip_module_prefix(name)
    return any(token in normalized_name for token in _RDE_ADAPTER_TRAINABLE_NAMES)


def apply_clip_trainability(model, args, logger):
    if getattr(args, "freeze_host", False):
        logger.info("--freeze_host active: keeping only target enrichment parameters trainable")
        return

    finetune_clip = _finetune_clip_enabled(args)
    for name, param in model.named_parameters():
        if _is_base_model_parameter(name):
            param.requires_grad_(finetune_clip or _is_rde_adapter_parameter(name))
        else:
            param.requires_grad_(True)

    total, trainable = _count_parameters(model)
    clip_total, clip_trainable = _count_parameters(model, _is_base_model_parameter)
    logger.info(
        "CLIP backbone fine-tuning: {}".format("enabled" if finetune_clip else "disabled")
    )
    logger.info(
        "CLIP backbone params: total=%s (%.3fM), trainable=%s (%.3fM), frozen=%s (%.3fM)",
        f"{clip_total:,}",
        clip_total / 1_000_000.0,
        f"{clip_trainable:,}",
        clip_trainable / 1_000_000.0,
        f"{clip_total - clip_trainable:,}",
        (clip_total - clip_trainable) / 1_000_000.0,
    )
    logger.info(
        "Trainable params after CLIP policy: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total if total else 0.0,
    )


def _cleanup_checkpoints_after_run(args, wandb_run, checkpoint_upload_complete, logger):
    if not getattr(args, "delete_checkpoints_after_run", False):
        return

    if getattr(args, "use_wandb", True) and not checkpoint_upload_complete:
        if wandb_run is None:
            logger.warning(
                "Checkpoint cleanup skipped: W&B was enabled but no active run was available "
                "to upload checkpoints."
            )
        else:
            logger.warning(
                "Checkpoint cleanup skipped: W&B checkpoint upload did not complete successfully."
            )
        return

    delete_output_checkpoints(args.output_dir, logger=logger)


if __name__ == '__main__':

    args = get_args()
    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
        warn_only=args.deterministic_warn_only,
    )
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}_{args.loss_names}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)
    if not os.path.isdir(args.output_dir + '/img'):
        os.makedirs(args.output_dir + '/img')
    wandb_run = None

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)

    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.to(device)
    finetune_checkpoint = _finetune_clip_checkpoint(args)
    if finetune_checkpoint:
        load_finetune_clip_checkpoint(model, finetune_checkpoint, logger)
    apply_clip_trainability(model, args, logger)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader, args)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
        logger.info(f"===================>start {start_epoch}")


    target_pool = None
    if getattr(args, "target_enrichment", False):
        target_pool = TargetPoolManager(train_loader.dataset, args, logger)

    if get_rank() == 0 and args.training:
        wandb_run = init_wandb(args, run_name=cur_time, output_dir=args.output_dir, logger=logger)

    run_completed = False
    checkpoint_upload_complete = not getattr(args, "use_wandb", True)
    try:
        do_train(
            start_epoch,
            args,
            model,
            train_loader,
            evaluator,
            optimizer,
            scheduler,
            checkpointer,
            target_pool,
            wandb_run=wandb_run,
        )
        if get_rank() == 0 and args.training:
            if getattr(args, "delete_checkpoints_after_run", False):
                if wandb_run is not None:
                    checkpoint_artifacts = upload_checkpoint_artifacts(
                        wandb_run,
                        args.output_dir,
                        logger=logger,
                    )
                    checkpoint_upload_complete = checkpoint_artifacts is not None
            else:
                upload_best_checkpoint_artifact(wandb_run, args.output_dir, logger=logger)
        run_completed = True
    finally:
        finish_wandb(wandb_run)
        if run_completed and get_rank() == 0 and args.training:
            _cleanup_checkpoints_after_run(
                args,
                wandb_run,
                checkpoint_upload_complete,
                logger,
            )
