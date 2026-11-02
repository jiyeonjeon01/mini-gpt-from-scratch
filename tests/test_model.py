"""Mini GPT 모델 테스트: Shape, Causal Mask, Pre-LN/Residual, Loss, 경계 조건, 파라미터 수"""

import copy

import pytest
import torch

from config import GPTConfig
from model import CausalSelfAttention, FeedForward, MiniGPT, TransformerBlock

V = 65  # Tiny Shakespeare의 실제 Vocabulary 크기
B, T = 2, 16


@pytest.fixture
def config():
    return GPTConfig(vocab_size=V)


@pytest.fixture
def model(config):
    torch.manual_seed(1337)
    return MiniGPT(config).eval()


@pytest.fixture
def idx():
    g = torch.Generator().manual_seed(0)
    return torch.randint(V, (B, T), generator=g)


# ---- Shape ----


def test_embedding_shape(model, idx):
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T))
    assert x.shape == (B, T, 128)


def test_attention_shape(config):
    attn = CausalSelfAttention(config)
    assert attn(torch.randn(B, T, 128)).shape == (B, T, 128)


def test_multi_head_config(config):
    attn = CausalSelfAttention(config)
    assert attn.n_head == 4
    assert attn.head_dim == 32
    assert attn.c_attn.weight.shape == (384, 128)  # Q/K/V를 한 번에 계산
    # Causal Mask는 학습 파라미터도 아니고 state_dict에도 저장되지 않는다
    assert "mask" not in attn.state_dict()
    assert "mask" not in dict(attn.named_parameters())


def test_ffn_shape(config):
    ffn = FeedForward(config)
    x = torch.randn(B, T, 128)
    assert ffn.fc(x).shape == (B, T, 512)
    assert ffn(x).shape == (B, T, 128)


def test_model_output_shape(model, idx):
    logits, loss = model(idx)
    assert logits.shape == (B, T, V)
    assert loss is None


# ---- Causal Mask ----


@pytest.mark.parametrize("k", [1, 8, 15])
def test_causal_future_tokens_do_not_change_past(model, idx, k):
    """k번째 이후 토큰을 바꿔도 0 ~ k-1 위치의 Logits는 그대로여야 한다."""
    idx2 = idx.clone()
    idx2[:, k:] = (idx[:, k:] + 1) % V  # 미래 토큰을 모두 다른 값으로 교체
    with torch.no_grad():
        logits1, _ = model(idx)
        logits2, _ = model(idx2)
    assert torch.allclose(logits1[:, :k], logits2[:, :k], atol=1e-6)
    # 대조군: 바꾼 위치부터는 Logits가 달라져야 한다 (변경이 실제로 모델에 전달됐는지 확인)
    assert not torch.allclose(logits1[:, k:], logits2[:, k:], atol=1e-6)


@pytest.mark.parametrize("k", [1, 8, 15])
def test_causal_prefix_matches_full_sequence(model, idx, k):
    """앞 k개만 넣은 결과와, 전체를 넣은 결과의 앞 k개 위치가 같아야 한다."""
    with torch.no_grad():
        full, _ = model(idx)
        prefix, _ = model(idx[:, :k])
    assert torch.allclose(prefix, full[:, :k], atol=1e-5)


def test_causal_attends_to_past_positions(config):
    """과거 토큰도 참고해야 한다 (미래만 막고 과거까지 막으면 안 된다).

    Q, K를 0으로 두면 허용된 위치의 Score가 모두 같아져 Attention이 균등해지고,
    V와 출력 Projection을 항등 변환으로 두면 위치 t의 출력은 x[0..t]의 평균이 된다.
    """
    attn = CausalSelfAttention(config)
    C = config.n_embd
    with torch.no_grad():
        attn.c_attn.weight.zero_()
        attn.c_attn.bias.zero_()
        attn.c_attn.weight[2 * C :] = torch.eye(C)  # V = x (Q, K = 0)
        attn.c_proj.weight.copy_(torch.eye(C))
        attn.c_proj.bias.zero_()

    x = torch.randn(1, T, C, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = attn(x)
        expected = x.cumsum(dim=1) / torch.arange(1, T + 1).view(1, T, 1)  # 누적 평균
        assert torch.allclose(out, expected, atol=1e-6)

        # 첫 번째 토큰만 바꾸면 마지막 위치의 출력은 정확히 변경량 / T 만큼 달라진다
        x2 = x.clone()
        x2[:, 0] += 1.0
        diff = attn(x2)[:, -1] - out[:, -1]
    assert torch.allclose(diff, torch.full_like(diff, 1.0 / T), atol=1e-6)


def test_causal_gradient_is_zero_for_future_positions(config):
    """위치 t의 출력은 t보다 뒤에 있는 입력에 대해 Gradient가 정확히 0이어야 한다."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(config)
    for t in range(T):
        x = torch.randn(1, T, 128, requires_grad=True)
        attn(x)[0, t].sum().backward()
        assert torch.all(x.grad[0, t + 1 :] == 0), f"t={t}: 미래 위치에 Gradient가 흘렀다"
        assert torch.isfinite(x.grad[0, : t + 1]).all()


# ---- Transformer Block: Residual, Pre-LN ----


def test_block_residual_identity_when_sublayers_are_zero(config):
    """Attention·FFN의 출력 Linear를 0으로 만들면 Block은 입력을 그대로 내보내야 한다."""
    block = TransformerBlock(config)
    with torch.no_grad():
        for layer in (block.attn.c_proj, block.ffn.proj):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(B, T, 128)
    assert torch.equal(block(x), x)


def test_block_is_pre_ln(config):
    """Attention에는 LN1(X)가, FFN에는 LN2(H)가 입력되어야 한다 (Pre-LN)."""
    torch.manual_seed(0)
    block = TransformerBlock(config).eval()
    captured = {}
    block.attn.register_forward_hook(lambda m, inp, out: captured.update(attn_in=inp[0]))
    block.ffn.register_forward_hook(lambda m, inp, out: captured.update(ffn_in=inp[0]))

    x = torch.randn(B, T, 128) * 3 + 1  # 정규화되지 않은 입력
    with torch.no_grad():
        y = block(x)
        h = x + block.attn(block.ln_1(x))
        expected_y = h + block.ffn(block.ln_2(h))
    assert torch.allclose(captured["attn_in"], block.ln_1(x).detach())
    assert torch.allclose(captured["ffn_in"], block.ln_2(h))
    assert torch.allclose(y, expected_y)


def test_blocks_are_independent_instances(model):
    blocks = list(model.blocks)
    assert len(blocks) == 4
    assert len({id(b) for b in blocks}) == 4
    ptrs = [b.attn.c_attn.weight.data_ptr() for b in blocks]
    assert len(set(ptrs)) == 4  # 가중치를 공유하지 않는다


# ---- Loss, Backward ----


def test_loss_is_finite_scalar(model, idx):
    targets = torch.roll(idx, -1, dims=1)
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, V)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_produces_finite_gradients(model, idx):
    targets = torch.roll(idx, -1, dims=1)
    _, loss = model(idx, targets)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name}: grad 없음"
        # T=16이면 Position Embedding의 16 ~ 127행은 사용되지 않아 0인 것이 정상이다
        assert torch.isfinite(p.grad).all(), f"{name}: grad에 NaN/Inf"


# ---- 경계 조건과 입력 검증 ----


def test_block_size_boundary(model):
    ok = torch.zeros(1, 128, dtype=torch.long)
    logits, _ = model(ok)
    assert logits.shape == (1, 128, V)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros(1, 129, dtype=torch.long))


@pytest.mark.parametrize(
    "bad_idx",
    [
        torch.zeros(16, dtype=torch.long),  # 1차원
        torch.zeros(2, 16),  # float
        torch.zeros(2, 0, dtype=torch.long),  # 길이 0
    ],
)
def test_invalid_idx_raises(model, bad_idx):
    with pytest.raises(ValueError):
        model(bad_idx)


@pytest.mark.parametrize(
    "bad_targets",
    [
        torch.zeros(2, 15, dtype=torch.long),  # Shape 불일치
        torch.zeros(2, 16),  # float
    ],
)
def test_invalid_targets_raises(model, idx, bad_targets):
    with pytest.raises(ValueError, match="targets"):
        model(idx, bad_targets)


def test_vocab_size_required():
    with pytest.raises(ValueError, match="vocab_size"):
        MiniGPT(GPTConfig())


# ---- 파라미터 수와 초기화 ----


def test_parameter_count(model):
    """809,728 + 257 × V = 826,433 (V=65, bias 포함, 가중치 비공유)"""
    assert model.num_parameters() == 826_433


def test_weight_initialization(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            assert torch.all(m.bias == 0)
        elif isinstance(m, torch.nn.LayerNorm):
            assert torch.all(m.weight == 1) and torch.all(m.bias == 0)
    # 원소가 많은 가중치의 표준편차는 0.02에 가깝다
    for w in (model.blocks[0].attn.c_attn.weight, model.tok_emb.weight, model.pos_emb.weight):
        assert abs(w.std().item() - 0.02) < 0.002


# ---- CPU / GPU ----


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA를 사용할 수 없다")
def test_cuda_matches_cpu(model, idx):
    targets = torch.roll(idx, -1, dims=1)
    gpu_model = copy.deepcopy(model).to("cuda")
    with torch.no_grad():
        cpu_logits, _ = model(idx)
        gpu_logits, gpu_loss = gpu_model(idx.cuda(), targets.cuda())
    assert gpu_logits.device.type == "cuda" and gpu_loss.device.type == "cuda"
    assert torch.allclose(cpu_logits, gpu_logits.cpu(), atol=1e-4)
