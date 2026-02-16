"""蒸馏训练循环：支持 7 种方法的统一接口。"""

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

logger = logging.getLogger(__name__)

VALID_METHODS = [
    "standard_kd", "fitnets", "prakash_cka",
    "tinybert", "minilm", "informed_fitnets", "rcid",
]


@dataclass
class TrainConfig:
    """训练超参数。"""
    method_name: str = "rcid"
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
    """统一蒸馏训练器，支持 7 种方法。"""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: TrainConfig,
        loss_fn: nn.Module,
        teacher_imprints: dict[tuple[int, int], torch.Tensor] | None = None,
        device: str | None = None,
    ) -> None:
        assert config.method_name in VALID_METHODS, (
            f"Unknown method: {config.method_name!r}, valid: {VALID_METHODS}"
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config
        self.method = config.method_name

        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = student_model
        self.student.to(self.device)
        self.student.train()

        self.loss_fn = loss_fn
        self.loss_fn.to(self.device)
        # 优化器: 学生参数 + loss_fn 可学习参数（TinyBERT/FitNets 投影）
        params = list(self.student.parameters())
        lp = list(self.loss_fn.parameters())
        if lp:
            params += lp
            logger.info("Added %d loss_fn params to optimizer", len(lp))
        self.optimizer = AdamW(params, lr=config.lr)
        self.scaler = torch.amp.GradScaler(enabled=config.use_amp)
        self.state = TrainState()
        logger.info("Trainer: method=%s, device=%s", self.method, self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> TrainState:
        """完整训练循环。"""
        cfg = self.config
        sched = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs * len(train_loader),
        )
        for epoch in range(cfg.epochs):
            self.state.epoch = epoch
            avg_loss, avg_parts = self._train_epoch(train_loader, sched, epoch)
            self.state.train_losses.append(avg_loss)
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.state.val_losses.append(val_loss)
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")
            logger.info(
                "Epoch %d/%d: train=%.4f%s", epoch + 1, cfg.epochs,
                avg_loss,
                f", val={val_loss:.4f}" if val_loss is not None else "",
            )
            if cfg.use_wandb:
                _try_wandb_log({"epoch": epoch, "train_loss": avg_loss,
                                **avg_parts,
                                **({"val_loss": val_loss} if val_loss else {})})
        self._save_checkpoint("final.pt")
        return self.state

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """验证集平均损失。"""
        self.student.eval()
        total, count = 0.0, 0
        for batch in val_loader:
            loss, _ = self._compute_batch_loss(batch)
            total += loss.item()
            count += 1
        self.student.train()
        return total / max(count, 1)

    def _train_epoch(
        self, loader: DataLoader, sched: CosineAnnealingLR, epoch: int,
    ) -> tuple[float, dict[str, float]]:
        self.student.train()
        total_loss, n = 0.0, 0
        accum: dict[str, float] = {}
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            lv, parts = self._train_step(batch)
            total_loss += lv
            for k, v in parts.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1
            self.state.global_step += 1
            sched.step()
            if self.state.global_step % self.config.log_every == 0:
                pbar.set_postfix(loss=f"{lv:.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")
        n = max(n, 1)
        return total_loss / n, {k: v / n for k, v in accum.items()}

    def _train_step(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[float, dict[str, float]]:
        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type=self.device.type,
                                enabled=self.config.use_amp):
            loss, parts = self._compute_batch_loss(batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(
            self.student.parameters(), self.config.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item(), {k: v.item() for k, v in parts.items()}

    def _compute_batch_loss(
        self, batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回 (total_loss, {部分损失名: detached 值})。"""
        clean = batch["clean_ids"].to(self.device)  # (B, S)
        m = self.method

        # TinyBERT / MiniLM: 自包含（KL+aux 都在 loss_fn 内）
        if m in ("tinybert", "minilm"):
            r = self.loss_fn(self.teacher, self.student, clean)
            return r["loss"], {k: v for k, v in r.items() if k != "loss"}

        # standard_kd: 仅 KL
        if m == "standard_kd":
            with torch.no_grad():
                tl = self.teacher(clean).logits     # (B, S, V)
            sl = self.student(clean).logits          # (B, S, V)
            kl = self.loss_fn(tl, sl)
            return kl, {"kl_loss": kl.detach()}

        # fitnets, prakash_cka, rcid, informed_fitnets
        parts: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=self.device)

        if self.config.lambda_kl > 0:
            kl = self._kl_loss(clean)
            total = total + self.config.lambda_kl * kl
            parts["kl_loss"] = kl.detach()

        aux, w = self._dispatch_aux(batch, clean, m)
        total = total + w * aux
        parts["aux_loss"] = aux.detach()
        return total, parts

    def _dispatch_aux(
        self, batch: dict[str, torch.Tensor],
        clean: torch.Tensor, method: str,
    ) -> tuple[torch.Tensor, float]:
        """分派辅助损失。返回 (loss, weight)。"""
        if method == "rcid":
            corrupt = batch["corrupt_ids"].to(self.device)
            imps = self._get_batch_imprints(clean, corrupt)
            return self.loss_fn(
                imps, self.student, clean, corrupt,
            ), self.config.lambda_rcid
        if method in ("fitnets", "prakash_cka"):
            t_res = self._get_teacher_residuals(
                clean, self.loss_fn._teacher_layers,
            )
            return self.loss_fn(t_res, self.student, clean), 1.0
        if method == "informed_fitnets":
            imps = self._get_batch_imprints(clean)
            return self.loss_fn(
                imps, self.student, clean,
            ), self.config.lambda_rcid
        return torch.tensor(0.0, device=self.device), 0.0

    @torch.no_grad()
    def _get_batch_imprints(
        self, clean: torch.Tensor, corrupt: torch.Tensor | None = None,
    ) -> dict[tuple[int, int], torch.Tensor]:
        """在线计算教师痕迹，自动跳过越界位置（短 batch）。"""
        cps = self.loss_fn.checkpoints
        layers = sorted(set(l for l, _ in cps))
        rc = self._get_teacher_residuals(clean, layers)
        seq_len = rc[layers[0]].shape[1]  # 当前 batch 实际 seq_len
        out: dict[tuple[int, int], torch.Tensor] = {}
        if corrupt is not None:
            rp = self._get_teacher_residuals(corrupt, layers)
            for l, t in cps:
                if t >= seq_len:
                    continue
                out[(l, t)] = rc[l][:, t, :] - rp[l][:, t, :]  # (B, d_T)
        else:
            for l, t in cps:
                if t >= seq_len:
                    continue
                out[(l, t)] = rc[l][:, t, :]  # (B, d_T)
        return out

    def _kl_loss(self, clean_ids: torch.Tensor) -> torch.Tensor:
        """通用 KL 损失。"""
        with torch.no_grad():
            tl = self.teacher(clean_ids).logits            # (B, S, V)
        sl = self.student(clean_ids).logits                 # (B, S, V)
        tau = 4.0
        tp = torch.softmax(tl / tau, dim=-1)               # (B, S, V)
        slp = torch.log_softmax(sl / tau, dim=-1)          # (B, S, V)
        return nn.functional.kl_div(slp, tp, reduction="batchmean") * tau**2

    @torch.no_grad()
    def _get_teacher_residuals(
        self, input_ids: torch.Tensor, layers: list[int],
    ) -> dict[int, torch.Tensor]:
        residuals: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            def _mh(l: int):  # noqa: E741
                def hook(mod, inp, out):
                    residuals[l] = out[0].detach()
                return hook
            handles.append(
                self.teacher.transformer.h[layer].register_forward_hook(_mh(layer))
            )
        try:
            self.teacher(input_ids)
        finally:
            for h in handles:
                h.remove()
        return residuals

    def _save_checkpoint(self, name: str) -> None:
        d = Path(self.config.save_dir); d.mkdir(parents=True, exist_ok=True)
        torch.save({"student_state_dict": self.student.state_dict(),
                     "optimizer_state_dict": self.optimizer.state_dict(),
                     "state": self.state}, d / name)


def _try_wandb_log(data: dict) -> None:
    try:
        import wandb; wandb.log(data)
    except Exception:
        pass
