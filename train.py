"""Mini GPT 학습: AdamW, Train/Validation Loss 기록, Checkpoint 저장과 재개

    uv run python train.py                                   # 본 학습 (TrainConfig 기본값)
    uv run python train.py --max-steps 10 --out-dir checkpoints/smoke --log-file logs/smoke.csv
    uv run python train.py --resume checkpoints/last.pt --max-steps 3000
"""

import argparse
import csv
import dataclasses
import hashlib
import math
import os
import time
import warnings
from pathlib import Path

import torch

from config import GPTConfig, TrainConfig, get_device, set_seed
from data import download_data, get_batch, load_text, split_data
from generate import CHECKPOINT_VERSION, generate_text, load_checkpoint, model_from_checkpoint
from model import MiniGPT
from tokenizer import CharTokenizer

# 학습 경로에 영향을 주는 설정: Resume 시 Checkpoint와 같아야 한다
# (max_steps, eval_*, 저장 경로는 학습 결과를 바꾸지 않으므로 달라도 된다)
RESUME_KEYS = (
    "learning_rate", "weight_decay", "beta1", "beta2",
    "batch_size", "grad_clip", "seed", "train_ratio",
)
CHECKPOINT_NAMES = ("init.pt", "best.pt", "last.pt")

# 학습 전후 비교용 생성 조건 (generate.py의 기본값과 같다)
SAMPLE_PROMPT = "ROMEO:"
SAMPLE_SEED = 1337


# ---- 데이터 ----


def load_data(cfg: TrainConfig) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """텍스트 다운로드 → 토크나이저 → 토큰 Tensor → Train/Validation 분리 (한 번만 수행)"""
    download_data(cfg.data_path)
    text = load_text(cfg.data_path)
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(cfg.vocab_path)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data, cfg.train_ratio)
    return tokenizer, train_data, val_data


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def data_info(tokenizer: CharTokenizer, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    """Checkpoint와 같은 데이터인지 확인하기 위한 정보 (Vocabulary, 길이, 내용 해시)"""
    return {
        "vocab": list(tokenizer.chars),
        "data_sizes": [len(train_data), len(val_data)],
        "data_sha256": [_sha256(train_data), _sha256(val_data)],
    }


# ---- Optimizer, 평가 ----


def build_optimizer(model: MiniGPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """2차원 이상 파라미터(Linear·Embedding 가중치)에만 Weight Decay를 적용한다.
    bias와 LayerNorm(1차원)은 감쇠하지 않는다."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


@torch.no_grad()
def evaluate(model: MiniGPT, splits: dict, cfg: TrainConfig, device: str) -> dict:
    """각 데이터 구간에서 eval_batches개 배치의 평균 Loss를 계산한다.

    평가 배치는 매번 같은 seed로 뽑는다: 평가 시점끼리 비교할 수 있고, 학습 난수도 소비하지 않는다.
    """
    was_training = model.training
    model.eval()
    losses = {}
    for name, data in splits.items():
        gen = torch.Generator().manual_seed(cfg.seed + 1)
        total = 0.0
        for _ in range(cfg.eval_batches):
            x, y = get_batch(data, cfg.batch_size, model.config.block_size, device, gen)
            total += model(x, y)[1].item()
        losses[name] = total / cfg.eval_batches
    model.train(was_training)
    return losses


# ---- Checkpoint ----


def _config_dict(cfg: TrainConfig) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}


def rng_state(train_gen: torch.Generator) -> dict:
    return {
        "train_gen": train_gen.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(rng: dict) -> torch.Generator:
    """전역 CPU·CUDA RNG를 복원하고, 학습 배치 Generator를 돌려준다."""
    torch.set_rng_state(rng["torch_cpu"])
    cuda_states = rng["torch_cuda"]
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            warnings.warn("GPU 개수가 달라 CUDA RNG 상태를 복원하지 않았다")
    train_gen = torch.Generator()
    train_gen.set_state(rng["train_gen"])
    return train_gen


def checkpoint_state(step, model, optimizer, cfg, info, losses, best_val_loss, train_gen) -> dict:
    """weights_only=True로 읽을 수 있도록 텐서와 기본 자료형만 담는다."""
    return {
        "format_version": CHECKPOINT_VERSION,
        "step": step,  # 이 Step까지 가중치 업데이트를 마친 상태
        "torch_version": str(torch.__version__),  # TorchVersion 객체는 weights_only로 읽을 수 없다
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": dataclasses.asdict(model.config),
        "train_config": _config_dict(cfg),
        **info,  # vocab, data_sizes, data_sha256
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "best_val_loss": best_val_loss,
        "rng": rng_state(train_gen),
    }


def save_checkpoint(path: Path, state: dict) -> None:
    """임시 파일에 저장한 뒤 교체한다. 저장 중 실패해도 기존 파일은 그대로 남는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def check_resume(ckpt: dict, cfg: TrainConfig, model_cfg: GPTConfig, info: dict,
                 allow_config_change: bool = False) -> None:
    """Checkpoint와 현재 데이터·모델·학습 설정이 같은지 확인한다."""
    # 데이터와 모델 구조는 어떤 옵션으로도 바꿀 수 없다
    for key in ("vocab", "data_sizes", "data_sha256"):
        if ckpt[key] != info[key]:
            raise ValueError(f"학습 데이터가 Checkpoint와 다르다 ({key}). 같은 데이터로만 재개할 수 있다")
    if ckpt["model_config"] != dataclasses.asdict(model_cfg):
        raise ValueError(f"모델 설정이 Checkpoint와 다르다: {ckpt['model_config']}")

    saved = ckpt["train_config"]
    diffs = [f"{k}: {saved.get(k)} → {getattr(cfg, k)}" for k in RESUME_KEYS if saved.get(k) != getattr(cfg, k)]
    if diffs:
        message = "학습 설정이 Checkpoint와 다르다: " + ", ".join(diffs)
        if not allow_config_change:
            raise ValueError(message + " (의도한 변경이면 --allow-config-change)")
        warnings.warn(message)


def check_log_for_resume(log_path: Path, step: int) -> None:
    """재개할 Step보다 뒤의 기록이 로그에 있으면 중단한다 (기존 기록은 지우지 않는다)."""
    if not log_path.exists():
        return
    with open(log_path, newline="", encoding="utf-8") as f:
        later = [int(row["step"]) for row in csv.DictReader(f) if int(row["step"]) > step]
    if later:
        raise ValueError(
            f"로그 {log_path}에 Checkpoint step {step} 이후의 기록(step {later[0]} ~ {later[-1]})이 있다. "
            "기존 기록은 보존하므로, --log-file로 새 로그 경로를 지정해 재개하라"
        )


def check_fresh_start(cfg: TrainConfig) -> None:
    """새 학습이 이전 결과를 덮어쓰지 않도록 확인한다."""
    out_dir = Path(cfg.checkpoint_dir)
    existing = [name for name in CHECKPOINT_NAMES if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"{out_dir}에 이미 Checkpoint가 있다 ({', '.join(existing)}). "
            "새 학습은 다른 --out-dir을 쓰고, 이어서 학습하려면 --resume을 사용하라"
        )
    log_path = Path(cfg.log_path)
    if log_path.exists() and log_path.stat().st_size > 0:
        raise FileExistsError(f"로그 {log_path}가 이미 있다. 다른 --log-file을 지정하라")


def restore_training(ckpt: dict, cfg: TrainConfig, device: str):
    """모델 → Optimizer → RNG 순서로 복원한다. 반환: model, optimizer, train_gen"""
    model = model_from_checkpoint(ckpt, device)  # 모델 생성(초기화)이 전역 RNG를 소비한다
    optimizer = build_optimizer(model, cfg)  # 모델을 device로 옮긴 뒤에 만든다
    optimizer.load_state_dict(ckpt["optimizer"])  # 상태는 파라미터의 device로 옮겨진다
    # 저장된 param_groups 대신 현재 설정을 적용한다 (--allow-config-change로 바꾼 값 반영)
    for group, weight_decay in zip(optimizer.param_groups, (cfg.weight_decay, 0.0)):
        group.update(lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=weight_decay)
    train_gen = restore_rng(ckpt["rng"])  # RNG는 마지막에 복원한다
    return model, optimizer, train_gen


# ---- 학습 ----


def append_log(log_path: Path, row: dict) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train(model, optimizer, train_data, val_data, cfg: TrainConfig, device: str,
          train_gen: torch.Generator, info: dict, start_step: int = 0,
          best_val_loss: float = math.inf, resumed: bool = False) -> float:
    """start_step부터 cfg.max_steps까지 학습한다. 반환: 가장 낮은 Validation Loss"""
    out_dir = Path(cfg.checkpoint_dir)
    splits = {"train": train_data, "val": val_data}
    grad_norm = None
    train_seconds, timed_steps = 0.0, 0

    model.train()
    for step in range(start_step, cfg.max_steps + 1):
        # 1. 평가와 저장: eval_interval마다, 그리고 마지막 Step에서
        #    재개한 Step은 Checkpoint를 저장할 때 이미 평가했으므로 건너뛴다
        is_eval_step = step % cfg.eval_interval == 0 or step == cfg.max_steps
        if is_eval_step and not (resumed and step == start_step):
            losses = evaluate(model, splits, cfg, device)
            ms_per_step = 1000 * train_seconds / timed_steps if timed_steps else None
            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]

            state = checkpoint_state(step, model, optimizer, cfg, info, losses, best_val_loss, train_gen)
            if step == 0 and not resumed:
                save_checkpoint(out_dir / "init.pt", state)
            save_checkpoint(out_dir / "last.pt", state)
            if is_best:
                save_checkpoint(out_dir / "best.pt", state)

            append_log(cfg.log_path, {
                "step": step,
                "train_loss": f"{losses['train']:.4f}",
                "val_loss": f"{losses['val']:.4f}",
                "grad_norm": "" if grad_norm is None else f"{grad_norm:.4f}",
                "ms_per_step": "" if ms_per_step is None else f"{ms_per_step:.2f}",
            })
            line = f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
            if grad_norm is not None:
                line += f" | grad_norm {grad_norm:.3f} | {ms_per_step:.1f} ms/step"
            print(line + (" | best" if is_best else ""))
            train_seconds, timed_steps = 0.0, 0

        if step == cfg.max_steps:
            break

        # 2. 가중치 업데이트 한 번
        t0 = time.perf_counter()
        x, y = get_batch(train_data, cfg.batch_size, model.config.block_size, device, train_gen)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step {step}: loss가 {loss.item()}이 되어 학습을 중단한다")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        optimizer.step()
        train_seconds += time.perf_counter() - t0
        timed_steps += 1

    return best_val_loss


def _print_sample(title: str, model: MiniGPT, tokenizer: CharTokenizer, max_new_tokens: int) -> None:
    print(f"\n[{title}] prompt={SAMPLE_PROMPT!r}, max_new_tokens={max_new_tokens}, "
          f"temperature=1.0, top_k=20, seed={SAMPLE_SEED}")
    print("-" * 60)
    print(generate_text(model, tokenizer, SAMPLE_PROMPT, max_new_tokens, seed=SAMPLE_SEED))
    print("-" * 60)


def run(cfg: TrainConfig, tokenizer: CharTokenizer, train_data: torch.Tensor, val_data: torch.Tensor,
        device: str, model_cfg: GPTConfig | None = None, resume: str | Path | None = None,
        allow_config_change: bool = False, sample_tokens: int = 300) -> float:
    """새 학습 또는 재개를 준비하고 학습한다. sample_tokens > 0이면 학습 전후 샘플을 출력한다."""
    model_cfg = model_cfg or GPTConfig(vocab_size=tokenizer.vocab_size)
    info = data_info(tokenizer, train_data, val_data)
    out_dir = Path(cfg.checkpoint_dir)

    if resume is not None:
        resume = Path(resume)
        last = out_dir / "last.pt"
        if last.exists() and last.resolve() != resume.resolve():
            raise FileExistsError(
                f"{out_dir}의 last.pt가 아닌 Checkpoint에서 재개하면 기존 결과를 덮어쓴다. 다른 --out-dir을 지정하라"
            )
        ckpt = load_checkpoint(resume)
        check_resume(ckpt, cfg, model_cfg, info, allow_config_change)
        check_log_for_resume(Path(cfg.log_path), ckpt["step"])
        model, optimizer, train_gen = restore_training(ckpt, cfg, device)
        start_step, best_val_loss = ckpt["step"], ckpt["best_val_loss"]
        print(f"재개: {resume} (step {start_step}, best val {best_val_loss:.4f})")
    else:
        check_fresh_start(cfg)
        set_seed(cfg.seed)
        model = MiniGPT(model_cfg).to(device)
        optimizer = build_optimizer(model, cfg)
        train_gen = torch.Generator().manual_seed(cfg.seed)
        start_step, best_val_loss = 0, math.inf
        if sample_tokens:
            _print_sample("학습 전 (init)", model, tokenizer, sample_tokens)

    print(f"\ndevice: {device} | 파라미터 수: {model.num_parameters():,}")
    print(f"lr {cfg.learning_rate} | weight_decay {cfg.weight_decay} | betas ({cfg.beta1}, {cfg.beta2}) | "
          f"batch {cfg.batch_size} × block {model.config.block_size} | grad_clip {cfg.grad_clip} | seed {cfg.seed}")
    print(f"max_steps {cfg.max_steps} | eval 간격 {cfg.eval_interval} × {cfg.eval_batches}배치 | "
          f"저장 {out_dir} | 로그 {cfg.log_path}\n")

    start = time.perf_counter()
    best_val_loss = train(model, optimizer, train_data, val_data, cfg, device, train_gen, info,
                          start_step, best_val_loss, resumed=resume is not None)
    print(f"\n학습 시간: {time.perf_counter() - start:.1f}s | best val loss: {best_val_loss:.4f}")

    if sample_tokens:
        best = load_checkpoint(out_dir / "best.pt")
        _print_sample(f"학습 후 (best.pt, step {best['step']})",
                      model_from_checkpoint(best, device), tokenizer, sample_tokens)
    return best_val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini GPT 학습")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-batches", type=int)
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--out-dir", type=Path, dest="checkpoint_dir")
    parser.add_argument("--log-file", type=Path, dest="log_path")
    parser.add_argument("--resume", type=Path, help="이어서 학습할 Checkpoint (보통 last.pt)")
    parser.add_argument("--allow-config-change", action="store_true",
                        help="학습 설정이 Checkpoint와 달라도 재개한다 (데이터·모델 불일치는 허용하지 않음)")
    parser.add_argument("--sample-tokens", type=int, default=300, help="학습 전후 샘플 길이 (0이면 생략)")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k in {f.name for f in dataclasses.fields(TrainConfig)} and v is not None}
    cfg = dataclasses.replace(TrainConfig(), **overrides)

    device = get_device()
    tokenizer, train_data, val_data = load_data(cfg)
    run(cfg, tokenizer, train_data, val_data, device, resume=args.resume,
        allow_config_change=args.allow_config_change, sample_tokens=args.sample_tokens)


if __name__ == "__main__":
    main()
