"""데이터 분리와 get_batch 테스트"""

import io
import urllib.request

import pytest
import torch

from config import PROJECT_ROOT, TrainConfig
from data import download_data, get_batch, split_data
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


def test_val_batch_comes_from_val_split():
    """검증 배치는 검증 구간 안의 토큰만 사용해야 한다 (학습 구간과 섞이지 않음)."""
    data = torch.arange(1000)
    train, val = split_data(data, train_ratio=0.9)  # train: 0~899, val: 900~999
    for _ in range(50):
        x, y = get_batch(val, batch_size=4, block_size=16)
        assert x.shape == y.shape == (4, 16)
        assert x.min().item() >= 900 and y.max().item() <= 999
        assert torch.equal(y, x + 1)


def test_config_paths_fixed_to_project_root():
    """실행 위치와 관계없이 데이터 경로가 프로젝트 루트의 data/ 를 가리켜야 한다."""
    cfg = TrainConfig()
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert cfg.data_path.is_absolute()
    assert cfg.data_path == PROJECT_ROOT / "data" / "input.txt"
    assert cfg.vocab_path == PROJECT_ROOT / "data" / "vocab.json"


# ---- 다운로드 (네트워크 없이 urlopen을 가짜 응답으로 바꿔서 테스트) ----


class FakeResponse(io.BytesIO):
    """urlopen 응답 흉내: fail_after 바이트를 보낸 뒤 연결이 끊긴다."""

    def __init__(self, body: bytes, content_length: int | None = None, fail_after=None):
        super().__init__(body)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}
        self.fail_after = fail_after

    def read(self, size=-1):
        if self.fail_after is not None:
            if self.tell() >= self.fail_after:
                raise ConnectionResetError("연결 끊김 (테스트)")
            remaining = self.fail_after - self.tell()
            size = remaining if size is None or size < 0 else min(size, remaining)
        return super().read(size)


def fake_urlopen(*responses):
    """호출될 때마다 responses를 차례대로 돌려주는 urlopen 대체 함수"""
    it = iter(responses)
    return lambda url, timeout=None: next(it)


BODY = b"First Citizen:\nBefore we proceed any further, hear me speak.\n"


def test_download_success(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen(FakeResponse(BODY, len(BODY))))
    path = tmp_path / "data" / "input.txt"
    download_data(path, url="http://example.invalid/input.txt")
    assert path.read_bytes() == BODY
    assert not (tmp_path / "data" / "input.txt.part").exists()


def test_download_interrupted_leaves_no_file(tmp_path, monkeypatch):
    """중간에 끊기면 최종 파일도, 임시 파일도 남지 않아야 한다."""
    monkeypatch.setattr(
        urllib.request, "urlopen", fake_urlopen(FakeResponse(BODY, len(BODY), fail_after=10))
    )
    path = tmp_path / "input.txt"
    with pytest.raises(ConnectionResetError):
        download_data(path, url="http://example.invalid/input.txt")
    assert not path.exists()
    assert not (tmp_path / "input.txt.part").exists()


def test_download_size_mismatch_rejected(tmp_path, monkeypatch):
    """서버가 알려준 크기보다 적게 받으면 불완전한 파일로 보고 버린다."""
    monkeypatch.setattr(
        urllib.request, "urlopen", fake_urlopen(FakeResponse(BODY[:10], len(BODY)))
    )
    path = tmp_path / "input.txt"
    with pytest.raises(OSError, match="크기 불일치"):
        download_data(path, url="http://example.invalid/input.txt")
    assert not path.exists()


def test_download_recovers_after_failure(tmp_path, monkeypatch):
    """실패한 뒤 다시 실행하면 처음부터 받아 완전한 파일을 만든다."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen(
            FakeResponse(BODY, len(BODY), fail_after=10),
            FakeResponse(BODY, len(BODY)),
        ),
    )
    path = tmp_path / "input.txt"
    with pytest.raises(ConnectionResetError):
        download_data(path, url="http://example.invalid/input.txt")
    download_data(path, url="http://example.invalid/input.txt")
    assert path.read_bytes() == BODY


def test_existing_file_is_not_downloaded_again(tmp_path, monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("파일이 이미 있으면 다운로드하지 않아야 한다")

    monkeypatch.setattr(urllib.request, "urlopen", must_not_be_called)
    path = tmp_path / "input.txt"
    path.write_bytes(BODY)
    download_data(path, url="http://example.invalid/input.txt")
    assert path.read_bytes() == BODY
