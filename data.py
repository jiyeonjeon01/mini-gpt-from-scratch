"""학습 데이터 준비: 다운로드, 토큰화, Train/Validation 분리, 배치 생성

실행하면 데이터를 준비하고 배치 하나를 만들어 Shape을 출력한다.
    uv run python data.py
"""

import os
import shutil
import urllib.request
from pathlib import Path

import torch

from config import GPTConfig, TrainConfig, get_device, set_seed
from tokenizer import CharTokenizer

# Tiny Shakespeare: 셰익스피어 희곡을 모은 약 1MB 영어 텍스트
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_data(path: str | Path, url: str = DATA_URL, timeout: float = 30) -> None:
    """파일이 없을 때만 학습 텍스트를 내려받는다.

    임시 파일(.part)에 먼저 받은 뒤, 끝까지 받았을 때만 최종 경로로 교체한다.
    중간에 실패하면 최종 파일이 만들어지지 않으므로, 다음 실행에서 다시 내려받는다.
    """
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".part")
    print(f"다운로드: {url}")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
            expected = resp.headers.get("Content-Length")
        # 서버가 알려준 크기와 실제로 받은 크기가 다르면 불완전한 파일이다
        if expected is not None and tmp_path.stat().st_size != int(expected):
            raise OSError(
                f"다운로드 크기 불일치: {tmp_path.stat().st_size} / {expected} bytes"
            )
        os.replace(tmp_path, path)  # 같은 디렉터리 안에서의 교체는 원자적이다
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_data(
    data: torch.Tensor, train_ratio: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """앞부분은 학습용, 뒷부분은 검증용으로 '연속'해서 나눈다.

    무작위로 섞어서 나누면 같은 문맥이 학습/검증 양쪽에 들어가
    검증 손실이 실제보다 좋게 나올 수 있다.
    """
    n = int(len(data) * train_ratio)
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """무작위 위치에서 길이 block_size인 (입력, 정답) 쌍을 batch_size개 만든다.

    정답 y는 입력 x를 한 칸 뒤로 민 것이다 (4편 '다음 토큰 예측').
        x: data[i     : i + block_size]
        y: data[i + 1 : i + block_size + 1]

    반환 Shape: x, y 모두 (batch_size, block_size) = (B, T)
    """
    if len(data) <= block_size:
        raise ValueError(
            f"데이터 길이({len(data)})가 block_size({block_size})보다 커야 한다"
        )
    # 시작 위치 i의 최댓값은 len(data) - block_size - 1 (y의 끝이 데이터 끝을 넘지 않도록)
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def main() -> None:
    train_cfg = TrainConfig()
    model_cfg = GPTConfig()
    set_seed(train_cfg.seed)
    device = get_device()

    # 1. 텍스트 준비
    download_data(train_cfg.data_path)
    text = load_text(train_cfg.data_path)
    print(f"전체 문자 수: {len(text):,}")

    # 2. 토크나이저 생성 (전체 텍스트 기준) 및 저장
    #    Vocabulary(문자 → ID 표)만 전체 텍스트에서 만든다. 검증 구간에만 있는 문자도
    #    ID를 받아 encode 오류가 나지 않게 하기 위함이다. 학습·검증의 정답 텍스트는
    #    아래 split_data에서 분리되므로 검증 문장이 학습에 쓰이지는 않는다.
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(train_cfg.vocab_path)
    print(f"Vocabulary 크기: {tokenizer.vocab_size}")
    print(f"Vocabulary: {''.join(tokenizer.chars)!r}")

    # 3. 전체 텍스트를 토큰 ID Tensor로 변환 후 분리
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(data, train_cfg.train_ratio)
    print(f"학습 토큰 수: {len(train_data):,} / 검증 토큰 수: {len(val_data):,}")
    val_only = set(text[len(train_data) :]) - set(text[: len(train_data)])
    print(f"검증 구간에만 있는 문자: {sorted(val_only)}")

    # 4. 배치 하나 생성 (학습용과 검증용 각각)
    x, y = get_batch(train_data, train_cfg.batch_size, model_cfg.block_size, device)
    vx, vy = get_batch(val_data, train_cfg.batch_size, model_cfg.block_size, device)
    print(f"\ndevice: {device}")
    print(f"x shape: {tuple(x.shape)}  (B, T)  dtype: {x.dtype}")
    print(f"y shape: {tuple(y.shape)}  (B, T)  dtype: {y.dtype}")
    print(f"검증 x/y shape: {tuple(vx.shape)} / {tuple(vy.shape)}")

    # 5. 첫 번째 시퀀스의 앞부분을 문자로 확인 (y가 x보다 한 칸 뒤인지)
    n = 40
    print(f"\nx[0] 앞 {n}자: {tokenizer.decode(x[0, :n])!r}")
    print(f"y[0] 앞 {n}자: {tokenizer.decode(y[0, :n])!r}")


if __name__ == "__main__":
    main()
