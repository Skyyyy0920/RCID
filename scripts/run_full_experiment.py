"""Full Experiment: 7 Methods x 5 Tasks x 3 Seeds = 105 runs.

调度脚本:
  a) 对所有任务提取因果痕迹（如果不存在）
  b) 对所有 (method, task, seed) 组合训练，跳过已完成的
  c) 训练完成后运行评估，汇总到 outputs/results/full_comparison.json

Usage: python scripts/run_full_experiment.py [--config configs/exp2_full_comparison.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcid.alignment.layer_matching import find_layer_mapping_from_models
from rcid.alignment.procrustes import procrustes_align
from rcid.circuit.checkpoint_selection import (
    checkpoints_to_tuples, collect_key_positions_docstring,
    collect_key_positions_greater_than, collect_key_positions_induction,
    collect_key_positions_ioi, collect_key_positions_sva, select_checkpoints,
)
from rcid.circuit.patching import extract_causal_imprints
from rcid.data.docstring import DocstringDataset
from rcid.data.greater_than import GreaterThanDataset
from rcid.data.induction import InductionDataset
from rcid.data.ioi import IOIDataset
from rcid.data.sva import SVADataset
from rcid.distillation.baselines import FitNetsLoss, PrakashCKALoss, StandardKDLoss
from rcid.distillation.informed_fitnets import InformedFitNetsLoss
from rcid.distillation.minilm import MiniLMStyleKD
from rcid.distillation.rcid_loss import RCIDLoss
from rcid.distillation.tinybert import TinyBERTStyleKD
from rcid.distillation.trainer import DistillationTrainer, TrainConfig
from rcid.eval import perplexity, task_accuracy
from rcid.eval.causal_consistency import CausalConsistencyEvaluator
from rcid.eval.ood_generators import OODTestGenerator
from rcid.eval.ood_robustness import RobustnessEvaluator
from rcid.models.student import StudentModel
from rcid.models.teacher import TeacherModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TASK_DATASETS = {
    "ioi": IOIDataset, "greater_than": GreaterThanDataset,
    "induction": InductionDataset, "sva": SVADataset, "docstring": DocstringDataset,
}
TASK_KEY_POS_FN = {
    "ioi": collect_key_positions_ioi, "greater_than": collect_key_positions_greater_than,
    "induction": collect_key_positions_induction, "sva": collect_key_positions_sva,
    "docstring": collect_key_positions_docstring,
}
# OOD 评估只对有 OOD 生成器的任务（docstring 暂无）
OOD_SUPPORTED_TASKS = {"ioi", "greater_than", "induction", "sva"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full 7x5x3 experiment")
    p.add_argument("--config", default="configs/exp2_full_comparison.yaml")
    p.add_argument("--dry-run", action="store_true", help="Print plan only")
    return p.parse_args()


def _make_student(cfg: DictConfig) -> StudentModel:
    """根据 cfg.student.init 创建学生模型。

    init="scratch": 从零随机初始化（使用 n_layers/d_model/n_head）。
    init=其他值:   视为 HuggingFace model ID 加载预训练权重（如 "distilgpt2"）。
    """
    init = cfg.student.get("init", "scratch")
    if init == "scratch":
        return StudentModel.from_scratch(
            n_layer=cfg.student.n_layers, n_embd=cfg.student.d_model,
            n_head=cfg.student.n_head,
        )
    return StudentModel.from_pretrained(init)


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path)


def ckpt_path(cfg: DictConfig, method: str, task: str, seed: int) -> Path:
    return Path(cfg.training.save_dir) / task / method / f"seed_{seed}" / "final.pt"


# -- Phase A: 提取因果痕迹 + 层映射 --

def extract_task_data(
    cfg: DictConfig, task_cfg: DictConfig, teacher: TeacherModel,
) -> dict[str, Any]:
    """为一个任务提取检查点、痕迹、层映射。"""
    task = task_cfg.name
    logger.info("--- Extracting: %s ---", task)
    ds_cls = TASK_DATASETS[task]
    train_ds = ds_cls(n_samples=task_cfg.n_train_pairs, seed=42)
    full_batch = next(iter(
        train_ds.to_dataloader(batch_size=task_cfg.n_train_pairs, shuffle=False)
    ))
    clean = full_batch["clean_ids"].to(teacher.device)
    corrupt = full_batch["corrupt_ids"].to(teacher.device)
    key_pos = TASK_KEY_POS_FN[task](full_batch)
    cp_results = select_checkpoints(
        teacher.model, clean, corrupt,
        key_positions=key_pos, top_k=cfg.circuit.top_k_checkpoints,
    )
    checkpoints = checkpoints_to_tuples(cp_results)
    imprints = extract_causal_imprints(teacher.model, clean, corrupt, checkpoints)
    set_seed(42)
    tmp_s = _make_student(cfg)
    t_layers = sorted(set(l for l, t in checkpoints))
    t_pos = sorted(set(t for l, t in checkpoints))
    mapping, _ = find_layer_mapping_from_models(
        teacher.model, tmp_s.model, clean, corrupt,
        teacher_layers=t_layers, token_positions=t_pos,
    )
    del tmp_s
    logger.info("  %d checkpoints, mapping=%s", len(checkpoints), mapping)
    return {
        "checkpoints": checkpoints, "teacher_imprints": imprints,
        "layer_mapping": mapping, "clean_ids": clean, "corrupt_ids": corrupt,
    }


# -- Phase B: 构建 loss_fn（使用方法特定参数）--

def _get(mc: DictConfig, key: str, default: float) -> float:
    """安全获取方法配置中的可选参数。"""
    return float(mc.get(key, default)) if hasattr(mc, key) else default


def build_loss_fn(
    method: str, cfg: DictConfig, mc: DictConfig,
    td: dict[str, Any], student: StudentModel,
) -> tuple[torch.nn.Module, dict[tuple[int, int], torch.Tensor] | None]:
    d_T, d_S = cfg.teacher.d_model, student.d_model
    n_T, n_S = cfg.teacher.n_layers, student.n_layers
    h_T = cfg.teacher.n_head
    h_S = student.model.config.n_head
    cps, mapping = td["checkpoints"], td["layer_mapping"]
    imps = td["teacher_imprints"]

    if method == "standard_kd":
        return StandardKDLoss(temperature=cfg.training.temperature), None
    if method == "fitnets":
        return FitNetsLoss(d_T, d_S, FitNetsLoss.make_uniform_pairs(n_T, n_S)), None
    if method == "prakash_cka":
        return PrakashCKALoss(d_T, d_S, FitNetsLoss.make_uniform_pairs(n_T, n_S)), None
    if method == "tinybert":
        pairs = TinyBERTStyleKD.make_uniform_pairs(n_T, n_S)
        return TinyBERTStyleKD(
            d_T, d_S, h_T, h_S, pairs,
            alpha=_get(mc, "alpha", 1.0), beta=_get(mc, "beta", 1.0),
            gamma=_get(mc, "gamma", 1.0),
        ), None
    if method == "minilm":
        return MiniLMStyleKD(
            h_T, h_S, [(n_T - 1, n_S - 1)],
            temperature=_get(mc, "vr_temperature", 1.0),
            alpha=_get(mc, "alpha", 1.0), beta=_get(mc, "beta", 1.0),
        ), None

    # RCID / InformedFitNets: Procrustes W
    student.model.eval()
    s_cps = [(mapping[l], t) for l, t in cps]
    s_imps = extract_causal_imprints(
        student.model, td["clean_ids"], td["corrupt_ids"], s_cps,
    )
    student.model.train()
    src = torch.cat([s_imps[(mapping[l], t)] for l, t in cps])
    tgt = torch.cat([imps[(l, t)] for l, t in cps])
    W = procrustes_align(src, tgt)
    if method == "rcid":
        return RCIDLoss(W, cps, mapping), None
    if method == "informed_fitnets":
        return InformedFitNetsLoss(W, cps, mapping), None
    raise ValueError(f"Unknown method: {method}")


# -- Phase C: 训练 + 评估单个 (method, task, seed) --

def run_single(
    cfg: DictConfig, method: str, mc: DictConfig,
    task_cfg: DictConfig, seed: int,
    teacher: TeacherModel, td: dict[str, Any],
    baselines: dict[str, float] | None = None,
) -> dict[str, Any]:
    task = task_cfg.name
    cp = ckpt_path(cfg, method, task, seed)
    if cp.exists():
        logger.info("SKIP: %s/%s/seed_%d", method, task, seed)
        mp = cp.parent / "metrics.json"
        if mp.exists():
            return json.loads(mp.read_text())
        return {"skipped": True}

    set_seed(seed)
    student = _make_student(cfg)
    loss_fn, run_imps = build_loss_fn(method, cfg, mc, td, student)
    tcfg = TrainConfig(
        method_name=method, epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size, lr=cfg.training.lr,
        max_grad_norm=cfg.training.max_grad_norm,
        lambda_rcid=cfg.training.lambda_rcid, lambda_kl=cfg.training.lambda_kl,
        seed=seed, save_dir=str(cp.parent),
    )
    ds_cls = TASK_DATASETS[task]
    train_ds = ds_cls(n_samples=task_cfg.n_train_pairs, seed=seed)
    val_ds = ds_cls(n_samples=task_cfg.n_val_pairs, seed=seed + 1000)
    train_ld = train_ds.to_dataloader(batch_size=cfg.training.batch_size)
    val_ld = val_ds.to_dataloader(batch_size=cfg.training.batch_size, shuffle=False)

    trainer = DistillationTrainer(
        teacher_model=teacher.model, student_model=student.model,
        config=tcfg, loss_fn=loss_fn, teacher_imprints=run_imps,
    )
    state = trainer.train(train_ld, val_ld)

    # -- 评估 --
    eval_ld = val_ds.to_dataloader(batch_size=cfg.training.batch_size, shuffle=False)
    student.model.eval()
    metrics: dict[str, Any] = {
        "train_losses": state.train_losses, "val_losses": state.val_losses,
    }
    if baselines:
        metrics["teacher_accuracy"] = baselines.get("teacher", -1)
        metrics["baseline_accuracy"] = baselines.get("student_untrained", -1)
    acc = task_accuracy.evaluate(student.model, eval_ld, task=task)
    metrics["task_accuracy"] = acc.get("accuracy", 0.0)
    ppl = perplexity.evaluate(student.model, max_samples=cfg.eval.perplexity_max_samples)
    metrics["perplexity"] = ppl["perplexity"]

    # 因果一致性
    cce = CausalConsistencyEvaluator()
    cc = cce.evaluate(
        teacher.model, student.model, eval_ld,
        td["checkpoints"], td["layer_mapping"],
    )
    metrics["causal_consistency"] = cc["mean_correlation"]

    # OOD 鲁棒性（仅支持的任务）
    if task in OOD_SUPPORTED_TASKS:
        ood_gen = OODTestGenerator(n_samples=100, seed=7777)
        ood_ds = ood_gen.generate(task)
        rob = RobustnessEvaluator()
        rob_result = rob.evaluate(student.model, eval_ld, ood_ds, task=task)
        metrics["ood_robustness"] = {
            k: v for k, v in rob_result.items()
            if not isinstance(v, torch.Tensor)
        }

    save_json(metrics, cp.parent / "metrics.json")
    return metrics


# -- Main --

def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    methods = [m.method_name for m in cfg.methods]
    tasks = [t.name for t in cfg.tasks]
    seeds = list(cfg.experiment.seeds)
    total = len(methods) * len(tasks) * len(seeds)
    logger.info("Plan: %d methods x %d tasks x %d seeds = %d runs",
                len(methods), len(tasks), len(seeds), total)
    for m in methods:
        logger.info("  method: %s", m)
    for t in tasks:
        logger.info("  task:   %s", t)

    if args.dry_run:
        for m in methods:
            for t in tasks:
                for s in seeds:
                    st = "DONE" if ckpt_path(cfg, m, t, s).exists() else "TODO"
                    logger.info("  [%s] %s / %s / seed_%d", st, m, t, s)
        return

    teacher = TeacherModel(cfg.teacher.name)
    td_map: dict[str, dict[str, Any]] = {}
    for tc in cfg.tasks:
        td_map[tc.name] = extract_task_data(cfg, tc, teacher)

    # Phase A.5: 基准线评估（每个任务一次）
    baselines: dict[str, dict[str, float]] = {}
    for tc in cfg.tasks:
        task = tc.name
        eval_ds = TASK_DATASETS[task](n_samples=tc.n_val_pairs, seed=9999)
        eval_ld = eval_ds.to_dataloader(batch_size=cfg.training.batch_size, shuffle=False)
        t_acc = task_accuracy.evaluate(teacher.model, eval_ld, task=task)["accuracy"]
        raw_s = _make_student(cfg); raw_s.model.eval()
        raw_s.model.to(teacher.device)
        s_acc = task_accuracy.evaluate(raw_s.model, eval_ld, task=task)["accuracy"]
        del raw_s
        baselines[task] = {"teacher": t_acc, "student_untrained": s_acc}
        logger.info("BASELINE %s | teacher=%.4f, untrained_student=%.4f", task, t_acc, s_acc)

    all_results: dict[str, dict[str, dict[int, dict]]] = {}
    done, skipped, t0 = 0, 0, time.time()
    for mc in cfg.methods:
        method = mc.method_name
        all_results.setdefault(method, {})
        for tc in cfg.tasks:
            task = tc.name
            all_results[method].setdefault(task, {})
            for seed in seeds:
                logger.info("=== [%d/%d] %s / %s / seed_%d ===",
                            done + 1, total, method, task, seed)
                met = run_single(cfg, method, mc, tc, seed, teacher, td_map[task],
                                baselines.get(task))
                all_results[method][task][seed] = met
                if met.get("skipped"):
                    skipped += 1
                done += 1

    elapsed = (time.time() - t0) / 60
    logger.info("Completed: %d runs (skipped %d) in %.0f min", done, skipped, elapsed)
    save_json(all_results, out / "full_comparison.json")
    logger.info("Results: %s", out / "full_comparison.json")


if __name__ == "__main__":
    main()
