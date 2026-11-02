"""데이터 분리와 get_batch 테스트"""

import pytest
import torch

from data import get_batch, split_data
from tokenizer import CharTokenizer


def test_blog_xy_example():
    """4편(수정본) 예시: 'I love AI' 9개 토큰 → 입력 8개, 정답 8개"""
    tok = CharTokenizer.from_text("I love AI")
    data = torch.tensor(tok.encode("I love AI"), dtype=torch.long)
    # 길이 9, block_size 8이면 가능한 시작 위치는 0 하나뿐이다
    x, y = get_batch(data, batch_size=1, block_size=8)
    assert tok.decode(x[0]) == "I love A"
    assert tok.decode(y[0]) == " love AI"


def test_shape_and_dtype():
    # B와 T를 서로 다른 값으로 두어 두 차원이 뒤바뀌는 버그를 잡는다
    B, T = 4, 16
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=B, block_size=T)
    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_y_is_x_shifted_by_one():
    # 데이터가 0, 1, 2, ... 이면 정답은 항상 입력 + 1이어야 한다
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=8, block_size=32)
    assert torch.equal(y, x + 1)


def test_batch_stays_in_range():
    # 시작 위치가 끝까지 가도 y가 데이터 범위를 넘지 않아야 한다
    data = torch.arange(20)
    for _ in range(200):
        x, y = get_batch(data, batch_size=4, block_size=10)
        assert y.max().item() <= 19


def test_same_seed_same_batch():
    data = torch.arange(1000)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    x1, _ = get_batch(data, 4, 16, generator=g1)
    x2, _ = get_batch(data, 4, 16, generator=g2)
    assert torch.equal(x1, x2)


def test_split_is_contiguous():
    data = torch.arange(100)
    train, val = split_data(data, train_ratio=0.9)
    assert torch.equal(train, torch.arange(90))
    assert torch.equal(val, torch.arange(90, 100))


def test_data_too_short_raises():
    with pytest.raises(ValueError):
        get_batch(torch.arange(8), batch_size=1, block_size=8)
