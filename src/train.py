import argparse
import atexit
import itertools
import math
import os
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import time as timer

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.dataset import ContrastiveMNIST, get_testset, get_trainset
from src.ESA.model import ESA, ESAConfig
from src.loss import ContrastiveLoss
from src.metrics import mean_average_precision, pairwise_cosine
from src.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    ctx: nullcontext | torch.amp.autocast
    ptdtype: torch.dtype = torch.float16
    ddp: bool = False
    rank: int = 0
    master: bool = True
    device: str = "cpu"
    device_type: str = "cpu"
    world_size: int = 1
    local_rank: int | None = None


def inititalize_training() -> TrainingConfig:
    """Initialize training in distributed data parallelism mode or normal mode."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if device_type == "cpu":
        ptdtype = torch.float16
        ctx = nullcontext()
    else:
        ptdtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ddp = int(os.environ.get("RANK", -1)) != -1

    if not ddp:
        device = device_type
        return TrainingConfig(
            ctx=ctx,
            ptdtype=ptdtype,
            device=device,
            device_type=device_type,
        )

    init_process_group(backend="nccl")
    atexit.register(destroy_process_group)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    return TrainingConfig(
        ctx=ctx,
        ptdtype=ptdtype,
        ddp=True,
        rank=rank,
        master=rank == 0,
        device=device,
        device_type=device_type,
        world_size=world_size,
        local_rank=local_rank,
    )


@dataclass
class Checkpoint:
    """Checkpoint configuration."""

    path: Path  # Path to the checkpoint file
    seed: int  # Manual seed for reproducibility
    step = 0  # Total number of gradient updates (optimizer steps)
    epoch = 0  # Number of iterations over the full dataset (epochs)
    epoch_step = 0  # Number of batches processed in the current epoch
    best_val_score = -1.0

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Save model, optimizer and training state to a checkpoint file."""
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_score": self.best_val_score,
            "step": self.step + 1,
            "epoch": self.epoch,
            "epoch_step": self.epoch_step,
            "seed": self.seed,
        }
        logger.info(f"Saving checkpoint to {self.path}")
        torch.save(checkpoint, self.path)

    def load(self, map_location: str) -> tuple[dict, dict]:
        """Load model, optimizer and training state from a checkpoint file."""
        checkpoint = torch.load(
            self.path,
            map_location=map_location,
            weights_only=True,
        )

        # Load training state
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.epoch_step = checkpoint["epoch_step"]
        self.best_val_score = checkpoint["best_val_score"]
        self.seed = checkpoint["seed"]

        # Load model weights
        model_state = checkpoint["model_state"]
        unwanted_prefix = "_orig_mod."
        for k, _ in list(model_state.items()):
            if k.startswith(unwanted_prefix):
                model_state[k[len(unwanted_prefix) :]] = model_state.pop(k)

        # Load optimizer state
        optimizer_state = checkpoint["optimizer_state"]

        return model_state, optimizer_state


def load_data(
    conf: TrainingConfig,
    batch_size: int,
    n_views: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Load train and test data and create dataloaders.

    Args:
        conf (TrainingConfig): The training configuration.
        batch_size (int): Batch size for dataloading.
        n_views (int): N viezs for contrastive dataset.
        num_workers (int): Parallel workers for data prefetching.

    Returns:
        tuple[DataLoader, DataLoader]: Train and test dataloaders.
    """
    mnist_train, mnist_test = get_trainset(), get_testset()
    trainset = ContrastiveMNIST(mnist_train, n_views=n_views)
    testset = ContrastiveMNIST(mnist_test, n_views=1, block_size=trainset.block_size)

    sampler = DistributedSampler(
        trainset,
        num_replicas=conf.world_size,
        rank=conf.rank,
        shuffle=True,
        drop_last=False,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    sampler = DistributedSampler(
        testset,
        num_replicas=conf.world_size,
        rank=conf.rank,
        shuffle=True,
        drop_last=False,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size * n_views,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return trainloader, testloader


def configure_optimizer(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    *,
    fused: bool = False,
    verbose: bool = False,
) -> torch.optim.Optimizer:
    """Configure the AdamW optimizer for the model.

    Args:
        model (nn.Module): The model to be optimized.
        weight_decay (float): AdamW weight decay.
        learning_rate (float): Learning rate.
        betas (tuple[float, float]): AdamW betas.
        fused (bool): To use the fused optimizer. Defaults to False.
        verbose (bool, optional): To print optimizer configuration. Defaults to False.

    Returns:
        torch.optim.Optimizer: AdamW optimizer.
    """
    # Select params that require grad
    param_dict = dict(model.named_parameters())
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() > 1]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() <= 1]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    if verbose:
        logger.info(
            f"Number of decayed parameter tensors: {len(decay_params)},"
            f" with {num_decay_params:,} parameters.",
        )
        logger.info(
            f"Number of non-decayed parameter tensors: {len(nodecay_params)},"
            f" with {num_nodecay_params:,} parameters.",
        )

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=fused)


def trapezoidal_scheduler(
    it: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
) -> float:
    """Trapezoidal scheduler.

    Args:
        it (int): current step.
        total_steps (int): maximum number of steps including warmup and cooldown.
        warmup_steps (int): number of warmup steps.
        max_lr (float): maximum learning rate.

    Raises:
        ValueError: if current step is greater than maximum number of steps.
        ValueError: if warmup steps is greater than maximum number of steps.

    Returns:
        float: new learning rate.
    """
    cooldown_steps = round(0.2 * total_steps)
    if it > total_steps:
        msg = f"Current iteration {it} must be smaller than max steps {total_steps}"
        raise ValueError(msg)
    if warmup_steps > total_steps:
        msg = f"Warmup steps {warmup_steps} must be smaller than max steps {total_steps}"
        raise ValueError(msg)

    # 1. Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    # 2. Constant lr for a while
    if it < total_steps - cooldown_steps:
        return max_lr

    # 3. 1-sqrt cooldown
    decay_ratio = 1 - math.sqrt((it - (total_steps - cooldown_steps)) / cooldown_steps)
    return max_lr * decay_ratio


def get_batch(iterator: Iterator, conf: TrainingConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve the next batch from the iterator and prefetch it to device."""
    try:
        batch = next(iterator)
        if conf.device_type == "cuda":
            features, masks, labels = (
                batch[0].to(conf.device, non_blocking=True),
                batch[1].to(conf.device, non_blocking=True),
                batch[2].to(conf.device, non_blocking=True),
            )
        else:
            features, masks, labels = (
                batch[0].to(conf.device),
                batch[1].to(conf.device),
                batch[2].to(conf.device),
            )
    except StopIteration:
        return None
    return (features, masks, labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    conf: TrainingConfig,
) -> None:
    """Evaluate the model performance on the test dataloader."""
    model.eval()
    embeddings, labels = [], []
    for batch in dataloader:
        if conf.device_type == "cuda":
            features, masks, batch_labels = (
                batch[0].to(conf.device, non_blocking=True),
                batch[1].to(conf.device, non_blocking=True),
                batch[2].to(conf.device, non_blocking=True),
            )
        else:
            features, masks, batch_labels = (
                batch[0].to(conf.device),
                batch[1].to(conf.device),
                batch[2].to(conf.device),
            )

        bsz, n_views, num_edges, features_dim = features.shape
        with conf.ctx:
            embs = model(
                features.view(bsz, num_edges, features_dim),
                masks.view(bsz, num_edges, num_edges),
            )

            embeddings.append(embs)
            labels.append(batch_labels)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    similarities = pairwise_cosine(embeddings)
    map_r, precision = mean_average_precision(similarities, labels)

    model.train()
    return map_r, precision


if __name__ == "__main__":
    """Train a GNN using contrastive learning in a distributed settings."""
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument(
        "--layers",
        type=str,
        default="SMMMMS",
        help="A list of self 'S' and masked 'M' attention blocks for the encoder.",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=2,
        help="Number of heads in the multi-head attention blocks.",
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=6,  # node_features * 2 + edge_features = 3 * 2 + 0 = 6
        help="Dimension of hidden states embeddings.",
    )
    parser.add_argument(
        "--pool-seeds",
        type=int,
        default=32,
        help="Number of seed vectors for the pooling module.",
    )
    parser.add_argument(
        "--pool-layers",
        type=int,
        default=2,
        help="Number of self attention blocks in the pooling module.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout for regularization.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="To compile the model. Doesn't work yet.",
    )
    # Training
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4096,
        help=(
            "Stop training after this many steps. A step is a gradient update (optimizer step),"
            " so it can contain multiple batches if using ddp or gradient accumulation."
        ),
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=16,
        help="Batch size. If gradient_acc_steps > 1, this is the micro-batch size.",
    )
    parser.add_argument(
        "--n-views",
        type=int,
        default=4,
        help="Number of views per example. There will be BSZ * N_VIEWS samples per micro-batch.",
    )
    parser.add_argument(
        "--gradient-acc-steps",
        type=int,
        default=4,
        help=(
            "Number of micro-steps to accumulate gradients before updating model parameters. "
            "Implies that the effective batch size is batch_size * gradient_acc_steps * world_size."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloading workers per process.",
    )
    # Learning rate scheduler
    parser.add_argument(
        "--max-lr",
        type=float,
        default=1e-5,
        help="Maximum learning rate for the trapezoidal lr scheduler.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=256,
        help="Number of warmup steps for the trapezoidal lr scheduler.",
    )
    # Optimizer
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-1,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Beta2 for AdamW optimizer.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value.",
    )
    # Run arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If 'resume', start from the checkpoint in output_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/outputs",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=40,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        help="Number of steps between loggings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=35,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode.",
    )
    args = parser.parse_args()

    conf = inititalize_training()
    trainloader, testloader = load_data(
        conf,
        batch_size=args.batch_size,
        n_views=args.n_views,
        num_workers=args.num_workers,
    )

    if conf.master:
        msg = f"Per process: {len(trainloader)} train batches, {len(testloader)} test batches."
        logger.info(msg)

    if conf.ddp:
        # Wait for all processes to reach this point before proceeding
        torch.distributed.barrier()

    model = ESA(
        config=ESAConfig(
            layers=args.layers,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            seeds=args.pool_seeds,
            pool_layers=args.pool_layers,
        ),
    )

    # Number of parameters
    if conf.master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Number of parameters: {n_params}")

    # AdamW optimizer
    optimizer = configure_optimizer(
        model,
        learning_rate=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        fused=conf.device_type == "cuda",
        verbose=conf.master,
    )

    criterion = ContrastiveLoss()

    args.output_dir.mkdir(exist_ok=True)
    ckpt = Checkpoint(path=args.output_dir / "ckpt.pt", seed=args.seed)
    if args.resume:
        model_state, optimizer_state = ckpt.load(map_location=conf.device)

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    scaler = torch.amp.GradScaler("cuda", enabled=(conf.ptdtype == torch.float16))

    model.to(conf.device)

    if args.compile:
        model = torch.compile(model)

    if conf.ddp:
        model = DistributedDataParallel(model, device_ids=[conf.local_rank])

    # Start training
    if conf.master:
        approx_epochs = (args.max_steps - ckpt.step) / (len(trainloader) / args.gradient_acc_steps)
        logger.info(
            "\nTraining plan :\n"
            f"- Will do {args.max_steps - ckpt.step} steps. (~{round(approx_epochs, 2)} epochs)\n"
            f"- Starting from step {ckpt.step} = epoch {ckpt.epoch} - "
            f"batch {ckpt.epoch_step}/{len(trainloader)}",
        )

    # Raw model for saving (can't save ddp model directly)
    raw_model = model.module if conf.ddp else model

    # Set the seed for reproducibility.
    torch.manual_seed(ckpt.seed + ckpt.epoch)
    train_iterator = iter(trainloader)

    if ckpt.epoch_step > 0:
        # Jump to the current epoch step.
        _ = next(itertools.islice(train_iterator, ckpt.epoch_step - 1, ckpt.epoch_step))

    # Prefetch first batch
    batch = get_batch(train_iterator, conf=conf)

    local_step = 0
    while ckpt.step < args.max_steps:
        # Determine and set the learning rate for this iteration
        lr = trapezoidal_scheduler(
            it=local_step,
            total_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            max_lr=args.max_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t_0 = timer()

        # Loop for gradient accumulation, multiple epoch-steps contribute to one optimizer step
        for acc_step in range(args.gradient_acc_steps):
            # batch contains n_views * batch_size graphs (as edge features and mask)
            features, masks, labels = batch
            bsz, n_views, num_edges, features_dim = features.shape

            if conf.ddp:
                # Official way is to use model.no_sync() context manager, seen in nanogpt.
                model.require_backward_grad_sync = acc_step == args.gradient_acc_steps - 1

            with conf.ctx:
                out = model(
                    features.view(bsz * n_views, num_edges, features_dim),
                    masks.view(bsz * n_views, num_edges, num_edges),
                )

                out = out.view(args.batch_size, args.n_views, -1)  # (bsz, n_views, output_dim)

                # Scale the loss to account for gradient accumulation
                loss = criterion(out, labels=labels) / args.gradient_acc_steps

            # Prefetch next batch
            ckpt.epoch_step += 1
            batch = get_batch(train_iterator, conf=conf)
            if batch is None:
                # We reached the end of this epoch
                ckpt.epoch += 1
                ckpt.epoch_step = 0

                # Set the seed for reproducibility.
                torch.manual_seed(ckpt.seed + ckpt.epoch)
                train_iterator = iter(trainloader)

                batch = get_batch(train_iterator, conf=conf)

            # Backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # Clip the gradient
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        dt = timer() - t_0
        if (ckpt.step + 1) % args.log_interval == 0 and conf.master:
            # Get loss as float. NOTE: this is a CPU-GPU sync point
            lossf = loss.item() * args.gradient_acc_steps

            logger.info(
                f"Step {ckpt.step:5} (Epoch {ckpt.epoch:2}, batch {ckpt.epoch_step:5})"
                f": loss {lossf:7.3f}, time {dt*1000:7.2f}ms",
            )

        # Evaluation
        if (ckpt.step + 1) % args.eval_interval == 0:
            if conf.master:
                logger.info(f"Step {ckpt.step}: Evaluating model...")

            map_r, precision = evaluate(model, testloader, conf=conf)

            if conf.ddp:
                # Gather val score from all workers
                dist.all_reduce(map_r, op=dist.ReduceOp.AVG)
                dist.all_reduce(precision, op=dist.ReduceOp.AVG)

            # Logging and saving
            if conf.master:
                map_r, precision = map_r.item(), precision.item()
                logger.info(f"Step {ckpt.step}: MAP@K = {map_r:.4f}, Precision = {precision:.4f}.")

                if map_r > ckpt.best_val_score:
                    ckpt.best_val_score = map_r
                    ckpt.save(model=raw_model, optimizer=optimizer)

        ckpt.step += 1
        local_step += 1
