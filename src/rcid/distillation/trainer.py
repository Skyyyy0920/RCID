"""蒸馏训练循环：支持 RCID / StandardKD / FitNets / PrakashCKA 统一接口。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from rcid.distillation.baselines import StandardKDLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练超参数（对应 configs/base.yaml 的 training 字段）。"""

    epochs: int = 20
    batch_size: int = 32
    lr: float = 5e-5
    lambda_rcid: float = 1.0
    lambda_kl: float = 1.0
    max_grad_norm: float = 1.0
    seed: int = 42
    use_amp: bool = False
    use_wandb: bool = False
    save_dir: str = "outputs/checkpoints"
    log_every: int = 10


@dataclass
class TrainState:
    """训练状态追踪。"""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class DistillationTrainer:
    """统一蒸馏训练器。"""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: TrainConfig,
        kd_loss_fn: StandardKDLoss | None = None,
        aux_loss_fn: nn.Module | None = None,
        teacher_imprints: dict[tuple[int, int], torch.Tensor] | None = None,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config

        # 教师: eval + frozen
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 学生: train mode
        self.student = student_model
        self.student.to(self.device)
        self.student.train()

        self.kd_loss_fn = kd_loss_fn
        self.aux_loss_fn = aux_loss_fn
        if aux_loss_fn is not None:
            self.aux_loss_fn.to(self.device)
        self.teacher_imprints = teacher_imprints
        if teacher_imprints is not None:
            self.teacher_imprints = {
                k: v.to(self.device) for k, v in teacher_imprints.items()
            }

        # 优化器：学生参数 + 辅助损失的可学习参数（如 FitNets 投影）
        params = list(self.student.parameters())
        if aux_loss_fn is not None:
            params += list(aux_loss_fn.parameters())
        self.optimizer = AdamW(params, lr=config.lr)

        self.scaler = torch.amp.GradScaler(enabled=config.use_amp)
        self.state = TrainState()
        self._aux_type = _detect_aux_type(aux_loss_fn)

        logger.info(
            "Trainer: device=%s, aux=%s, lr=%.1e",
            self.device, self._aux_type, config.lr,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> TrainState:
        """完整训练循环。"""
        cfg = self.config
        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs * len(train_loader),
        )
        if cfg.use_wandb:
            _try_wandb_init(cfg)

        for epoch in range(cfg.epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch(train_loader, scheduler, epoch)
            self.state.train_losses.append(epoch_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.state.val_losses.append(val_loss)
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")

            logger.info(
                "Epoch %d/%d: train=%.4f%s", epoch + 1, cfg.epochs,
                epoch_loss,
                f", val={val_loss:.4f}" if val_loss is not None else "",
            )
            if cfg.use_wandb:
                _try_wandb_log({
                    "epoch": epoch, "train_loss": epoch_loss,
                    **({"val_loss": val_loss} if val_loss is not None else {}),
                })

        self._save_checkpoint("final.pt")
        return self.state

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """在验证集上评估，返回平均损失。"""
        self.student.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            loss = self._compute_batch_loss(batch)
            total += loss.item()
            count += 1
        self.student.train()
        return total / max(count, 1)

    def _train_epoch(
        self, loader: DataLoader, scheduler: CosineAnnealingLR, epoch: int,
    ) -> float:
        """单 epoch 训练，返回平均损失。"""
        self.student.train()
        total_loss, n_steps = 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            loss = self._train_step(batch)
            total_loss += loss
            n_steps += 1
            self.state.global_step += 1
            scheduler.step()

            if self.state.global_step % self.config.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                if self.config.use_wandb:
                    _try_wandb_log({
                        "step": self.state.global_step,
                        "step_loss": loss,
                        "lr": scheduler.get_last_lr()[0],
                    })

        return total_loss / max(n_steps, 1)

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """forward -> loss -> backward -> step，返回损失值。"""
        self.optimizer.zero_grad()
        amp_ctx = torch.amp.autocast(
            device_type=self.device.type, enabled=self.config.use_amp,
        )
        with amp_ctx:
            loss = self._compute_batch_loss(batch)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.student.parameters(), self.config.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def _compute_batch_loss(
        self, batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """根据方法类型计算总损失。batch: {clean_ids, corrupt_ids, ...}。"""
        cfg = self.config
        clean_ids = batch["clean_ids"].to(self.device)  # (B, seq_len)
        total_loss = torch.tensor(0.0, device=self.device)
        # KL 损失
        if self.kd_loss_fn is not None and cfg.lambda_kl > 0:
            with torch.no_grad():
                t_logits = self.teacher(clean_ids).logits   # (B, S, V)
            s_logits = self.student(clean_ids).logits        # (B, S, V)
            kl_loss = self.kd_loss_fn(t_logits, s_logits)
            total_loss = total_loss + cfg.lambda_kl * kl_loss
        # 辅助损失
        if self.aux_loss_fn is not None and self._aux_type != "none":
            aux = self._compute_aux_loss(batch, clean_ids)
            w = cfg.lambda_rcid if self._aux_type == "rcid" else 1.0
            total_loss = total_loss + w * aux

        return total_loss

    def _compute_aux_loss(
        self, batch: dict[str, torch.Tensor], clean_ids: torch.Tensor,
    ) -> torch.Tensor:
        """分派辅助损失计算。"""
        if self._aux_type == "rcid":
            corrupt_ids = batch["corrupt_ids"].to(self.device)
            return self.aux_loss_fn(
                self.teacher_imprints, self.student, clean_ids, corrupt_ids,
            )
        # FitNets / PrakashCKA: 需要教师残差流（每 batch 计算）
        teacher_layers = self.aux_loss_fn._teacher_layers
        teacher_res = self._get_teacher_residuals(clean_ids, teacher_layers)
        return self.aux_loss_fn(teacher_res, self.student, clean_ids)

    @torch.no_grad()
    def _get_teacher_residuals(
        self, input_ids: torch.Tensor, layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """提取教师残差流（no_grad, detached）。"""
        residuals: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            def _make_hook(l: int):  # noqa: E741
                def hook(mod, inp, out):
                    residuals[l] = out[0].detach()
                return hook
            handle = self.teacher.transformer.h[layer].register_forward_hook(
                _make_hook(layer),
            )
            handles.append(handle)

        try:
            self.teacher(input_ids)
        finally:
            for h in handles:
                h.remove()
        return residuals

    def _save_checkpoint(self, filename: str) -> None:
        """保存学生模型 checkpoint。"""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        torch.save({
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": self.state,
        }, path)
        logger.info("Saved checkpoint: %s", path)


def _detect_aux_type(loss_fn: nn.Module | None) -> str:
    """检测辅助损失类型。"""
    if loss_fn is None:
        return "none"
    name = type(loss_fn).__name__
    if "RCID" in name:
        return "rcid"
    if "FitNet" in name:
        return "fitnets"
    if "Prakash" in name or "CKA" in name:
        return "prakash"
    return "unknown"


def _try_wandb_init(cfg: TrainConfig) -> None:
    try:
        import wandb; wandb.init(project="rcid", config=vars(cfg))
    except Exception as e:
        logger.warning("wandb init failed: %s", e)


def _try_wandb_log(data: dict) -> None:
    try:
        import wandb; wandb.log(data)
    except Exception:
        pass
