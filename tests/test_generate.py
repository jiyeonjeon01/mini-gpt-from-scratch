"""텍스트 생성 테스트: 길이, Context 자르기, Top-k, Temperature, 재현성, 입력 검증"""

import pytest
import torch

from config import GPTConfig
from generate import generate, generate_text
from model import MiniGPT
from tokenizer import CharTokenizer

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
BLOCK = 16


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(TEXT)


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK, n_embd=32, n_head=2, n_layer=1, ffn_dim=64
    )
    return MiniGPT(cfg).eval()


def gen(seed=0, device="cpu"):
    return torch.Generator(device=device).manual_seed(seed)


def test_output_length(model):
    idx = torch.zeros(1, 3, dtype=torch.long)
    out = generate(model, idx, max_new_tokens=10, generator=gen())
    assert out.shape == (1, 13)
    assert torch.equal(out[:, :3], idx)  # 입력은 그대로 앞에 남는다


def test_prompt_longer_than_block_size(model):
    """block_size보다 긴 입력도 마지막 block_size개만 사용해 생성할 수 있다.
    (자르지 않으면 MiniGPT.forward가 ValueError를 낸다)"""
    idx = torch.zeros(1, BLOCK + 24, dtype=torch.long)
    out = generate(model, idx, max_new_tokens=5, generator=gen())
    assert out.shape == (1, BLOCK + 29)


def test_top_k_1_is_greedy(model):
    """top_k=1이면 항상 점수가 가장 높은 토큰을 고른다 (seed와 무관)."""
    idx = torch.zeros(1, 1, dtype=torch.long)
    expected = idx
    with torch.no_grad():
        for _ in range(20):
            logits, _ = model(expected[:, -BLOCK:])
            expected = torch.cat([expected, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    for seed in (0, 1, 2):
        assert torch.equal(generate(model, idx, 20, top_k=1, generator=gen(seed)), expected)


def test_top_k_restricts_candidates(model):
    idx = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        logits, _ = model(idx)
    allowed = set(torch.topk(logits[0, -1], 2).indices.tolist())
    sampled = {generate(model, idx, 1, top_k=2, generator=gen(s))[0, -1].item() for s in range(200)}
    assert sampled <= allowed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"top_k": 0},
        {"top_k": 1000},  # vocab_size보다 큼
        {"max_new_tokens": -1},
    ],
)
def test_invalid_options_raise(model, kwargs):
    args = {"max_new_tokens": 5, **kwargs}
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 1, dtype=torch.long), generator=gen(), **args)


def test_empty_prompt_raises(model, tokenizer):
    with pytest.raises(ValueError, match="prompt"):
        generate_text(model, tokenizer, "")


def test_unknown_prompt_char_raises(model, tokenizer):
    with pytest.raises(ValueError, match="Vocabulary에 없는 문자"):
        generate_text(model, tokenizer, "ROMEO: 안녕")


def test_tokenizer_model_vocab_mismatch_raises(model):
    with pytest.raises(ValueError, match="Vocabulary 크기"):
        generate_text(model, CharTokenizer.from_text("abc"), "abc")


def test_same_seed_same_text(model, tokenizer):
    a = generate_text(model, tokenizer, "First", max_new_tokens=30, seed=7)
    b = generate_text(model, tokenizer, "First", max_new_tokens=30, seed=7)
    assert a == b
    assert a.startswith("First") and len(a) == len("First") + 30


def test_restores_mode_and_no_grad(model, tokenizer):
    model.train()
    generate_text(model, tokenizer, "First", max_new_tokens=5)
    assert model.training
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA를 사용할 수 없다")
def test_generate_on_cuda(model, tokenizer):
    model = model.cuda()
    text = generate_text(model, tokenizer, "First", max_new_tokens=20, seed=1)
    assert text == generate_text(model, tokenizer, "First", max_new_tokens=20, seed=1)
    assert len(text) == len("First") + 20
