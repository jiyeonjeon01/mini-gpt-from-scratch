"""학습 테스트: 가중치 업데이트, 평가, Checkpoint, RNG, Resume, 설정·데이터·로그 검증

작은 합성 데이터와 작은 모델을 CPU에서 사용한다 (네트워크 없음).
"""

import csv
import dataclasses
import math

import pytest
import torch

from config import GPTConfig, TrainConfig
from data import split_data
from generate import load_checkpoint, model_from_checkpoint
from model import MiniGPT
from tokenizer import CharTokenizer
import train as train_module
from train import (
    build_optimizer,
    check_log_for_resume,
    check_resume,
    data_info,
    evaluate,
    restore_rng,
    rng_state,
    run,
    save_checkpoint,
)

TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\n"
) * 20


@pytest.fixture
def setup():
    tokenizer = CharTokenizer.from_text(TEXT)
    data = torch.tensor(tokenizer.encode(TEXT), dtype=torch.long)
    train_data, val_data = split_data(data, 0.9)
    model_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=16, n_embd=32, n_head=2, n_layer=1, ffn_dim=64
    )
    return tokenizer, train_data, val_data, model_cfg


def make_cfg(tmp_path, name="run", **kwargs):
    base = dict(
        checkpoint_dir=tmp_path / name, log_path=tmp_path / f"{name}.csv",
        batch_size=4, max_steps=10, eval_interval=5, eval_batches=2,
    )
    return TrainConfig(**{**base, **kwargs})


def run_tiny(setup, cfg, **kwargs):
    tokenizer, train_data, val_data, model_cfg = setup
    return run(cfg, tokenizer, train_data, val_data, "cpu", model_cfg, sample_tokens=0, **kwargs)


def params(path):
    return load_checkpoint(path)["model"]


def same_params(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


# ---- Optimizer, 학습 ----


def test_weight_decay_groups(setup):
    model = MiniGPT(setup[3])
    decay, no_decay = build_optimizer(model, TrainConfig()).param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    assert sum(p.numel() for g in (decay, no_decay) for p in g["params"]) == model.num_parameters()


def test_training_updates_weights(setup, tmp_path):
    cfg = make_cfg(tmp_path)
    run_tiny(setup, cfg)
    init, last = params(cfg.checkpoint_dir / "init.pt"), params(cfg.checkpoint_dir / "last.pt")
    assert load_checkpoint(cfg.checkpoint_dir / "last.pt")["step"] == 10
    assert not same_params(init, last)


def test_loss_decreases_on_fixed_batch(setup):
    """같은 배치를 반복 학습하면 Loss가 줄어야 한다 (특정 값은 검사하지 않는다)."""
    _, train_data, _, model_cfg = setup
    torch.manual_seed(0)
    model = MiniGPT(model_cfg)
    optimizer = build_optimizer(model, TrainConfig(learning_rate=1e-2))
    x, y = train_data[:16].view(1, 16), train_data[1:17].view(1, 16)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_nan_loss_stops_training(setup, tmp_path):
    tokenizer, train_data, val_data, model_cfg = setup
    model = MiniGPT(model_cfg)
    with torch.no_grad():
        model.lm_head.bias.fill_(float("nan"))
    cfg = make_cfg(tmp_path)
    with pytest.raises(FloatingPointError):
        train_module.train(model, build_optimizer(model, cfg), train_data, val_data, cfg, "cpu",
                           torch.Generator().manual_seed(0), data_info(tokenizer, train_data, val_data))


# ---- 평가 ----


def test_evaluate_does_not_change_model(setup):
    _, train_data, val_data, model_cfg = setup
    model = MiniGPT(model_cfg).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = TrainConfig(batch_size=4, eval_batches=3)
    splits = {"train": train_data, "val": val_data}
    first = evaluate(model, splits, cfg, "cpu")
    second = evaluate(model, splits, cfg, "cpu")
    assert first == second  # 매번 같은 평가 배치
    assert math.isfinite(first["train"]) and math.isfinite(first["val"])
    assert model.training  # 원래 모드로 복구
    assert all(p.grad is None for p in model.parameters())
    assert same_params(before, model.state_dict())


def test_eval_interval_does_not_change_training(setup, tmp_path):
    """평가는 학습 난수를 쓰지 않으므로, 평가 간격이 달라도 학습 결과가 같아야 한다."""
    a, b = make_cfg(tmp_path, "a", eval_interval=2), make_cfg(tmp_path, "b", eval_interval=5)
    run_tiny(setup, a)
    run_tiny(setup, b)
    assert same_params(params(a.checkpoint_dir / "last.pt"), params(b.checkpoint_dir / "last.pt"))


# ---- Checkpoint, RNG ----


def test_checkpoint_roundtrip(setup, tmp_path):
    cfg = make_cfg(tmp_path)
    run_tiny(setup, cfg)
    ckpt = load_checkpoint(cfg.checkpoint_dir / "last.pt")  # weights_only=True
    assert ckpt["vocab"] == setup[0].chars
    assert {"init.pt", "best.pt", "last.pt"} <= {p.name for p in cfg.checkpoint_dir.iterdir()}

    model = model_from_checkpoint(ckpt, "cpu").eval()
    reloaded = model_from_checkpoint(load_checkpoint(cfg.checkpoint_dir / "last.pt"), "cpu").eval()
    idx = setup[1][:16].view(1, 16)
    with torch.no_grad():
        assert torch.equal(model(idx)[0], reloaded(idx)[0])


def test_save_failure_keeps_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"value": 1})

    def broken_save(obj, f):
        open(f, "wb").write(b"partial")
        raise OSError("디스크 오류 (테스트)")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        save_checkpoint(path, {"value": 2})
    monkeypatch.undo()
    assert torch.load(path, weights_only=True) == {"value": 1}
    assert not (tmp_path / "last.pt.tmp").exists()


def test_rng_state_roundtrip():
    gen = torch.Generator().manual_seed(3)
    state = rng_state(gen)
    expected_global = torch.rand(3)
    expected_gen = torch.randint(100, (5,), generator=gen)
    restored = restore_rng(state)
    assert torch.equal(torch.rand(3), expected_global)
    assert torch.equal(torch.randint(100, (5,), generator=restored), expected_gen)


# ---- Resume ----


def test_resume_matches_continuous_training(setup, tmp_path):
    """연속 10 Step = 5 Step 학습 후 재개해 10 Step (CPU에서 동일해야 한다)."""
    continuous = make_cfg(tmp_path, "continuous", max_steps=10)
    run_tiny(setup, continuous)

    split = make_cfg(tmp_path, "split", max_steps=5)
    run_tiny(setup, split)
    resumed = dataclasses.replace(split, max_steps=10)
    run_tiny(setup, resumed, resume=split.checkpoint_dir / "last.pt")

    a = load_checkpoint(continuous.checkpoint_dir / "last.pt")
    b = load_checkpoint(split.checkpoint_dir / "last.pt")
    assert a["step"] == b["step"] == 10
    assert same_params(a["model"], b["model"])
    with open(split.log_path, newline="") as f:
        assert [int(r["step"]) for r in csv.DictReader(f)] == [0, 5, 10]  # 중복 기록 없음


def test_resume_rejects_different_data(setup, tmp_path):
    tokenizer, train_data, val_data, model_cfg = setup
    cfg = make_cfg(tmp_path)
    run_tiny(setup, cfg)
    ckpt = load_checkpoint(cfg.checkpoint_dir / "last.pt")

    changed = train_data.clone()
    changed[0], changed[1] = train_data[1], train_data[0]  # 문자 종류와 길이는 같고 내용만 다름
    info = data_info(tokenizer, changed, val_data)
    with pytest.raises(ValueError, match="data_sha256"):
        check_resume(ckpt, cfg, model_cfg, info, allow_config_change=True)  # 옵션으로도 우회 불가

    other_vocab = data_info(CharTokenizer.from_text(TEXT + "Z"), train_data, val_data)
    with pytest.raises(ValueError, match="vocab"):
        check_resume(ckpt, cfg, model_cfg, other_vocab, allow_config_change=True)


def test_resume_config_mismatch(setup, tmp_path):
    tokenizer, train_data, val_data, model_cfg = setup
    cfg = make_cfg(tmp_path)
    run_tiny(setup, cfg)
    ckpt = load_checkpoint(cfg.checkpoint_dir / "last.pt")
    info = data_info(tokenizer, train_data, val_data)

    changed = dataclasses.replace(cfg, learning_rate=3e-4)
    with pytest.raises(ValueError, match="learning_rate"):
        check_resume(ckpt, changed, model_cfg, info)
    with pytest.warns(UserWarning, match="learning_rate"):
        check_resume(ckpt, changed, model_cfg, info, allow_config_change=True)
    # 학습 경로에 영향이 없는 설정은 달라도 된다
    check_resume(ckpt, dataclasses.replace(cfg, max_steps=99, eval_interval=7), model_cfg, info)


def test_resume_rejects_log_with_later_steps(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("step,train_loss\n0,4.1\n5,3.0\n10,2.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="--log-file"):
        check_log_for_resume(log, 5)
    assert log.read_text(encoding="utf-8").count("\n") == 4  # 기존 기록은 그대로
    check_log_for_resume(log, 10)  # 이후 기록이 없으면 통과


def test_fresh_run_does_not_overwrite(setup, tmp_path):
    cfg = make_cfg(tmp_path)
    run_tiny(setup, cfg)
    with pytest.raises(FileExistsError):
        run_tiny(setup, cfg)
    with pytest.raises(FileExistsError, match="log"):
        run_tiny(setup, dataclasses.replace(cfg, checkpoint_dir=tmp_path / "new"))


def test_resume_from_non_last_checkpoint_rejected(setup, tmp_path):
    cfg = make_cfg(tmp_path)
    run_tiny(setup, cfg)
    with pytest.raises(FileExistsError, match="out-dir"):
        run_tiny(setup, cfg, resume=cfg.checkpoint_dir / "init.pt")


# ---- CUDA ----


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA를 사용할 수 없다")
def test_train_and_resume_on_cuda(setup, tmp_path):
    tokenizer, train_data, val_data, model_cfg = setup
    cfg = make_cfg(tmp_path, max_steps=4, eval_interval=2)
    run(cfg, tokenizer, train_data, val_data, "cuda", model_cfg, sample_tokens=0)
    resumed = dataclasses.replace(cfg, max_steps=6)
    run(resumed, tokenizer, train_data, val_data, "cuda", model_cfg,
        resume=cfg.checkpoint_dir / "last.pt", sample_tokens=0)
    ckpt = load_checkpoint(cfg.checkpoint_dir / "last.pt")
    assert ckpt["step"] == 6 and ckpt["rng"]["torch_cuda"] is not None
    assert next(model_from_checkpoint(ckpt, "cuda").parameters()).is_cuda
