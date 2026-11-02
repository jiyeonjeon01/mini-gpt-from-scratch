"""프로젝트 전체 설정과 공통 유틸리티 (Seed 고정, Device 선택)"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# 어느 디렉터리에서 실행해도 같은 data/ 를 쓰도록 프로젝트 루트(이 파일 위치) 기준으로 고정한다
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class GPTConfig:
    """모델 구조 설정 (4편 '6. 모델 설계'에서 정한 값)"""

    vocab_size: int | None = None  # 토크나이저를 만든 뒤 실제 문자 종류 수로 채운다
    block_size: int = 128  # 한 번에 참고할 수 있는 최대 토큰 수 (T)
    n_embd: int = 128  # 토큰 임베딩 차원 (C)
    n_head: int = 4  # Attention Head 개수
    n_layer: int = 4  # Transformer Block 개수
    ffn_dim: int = 512  # FFN 내부 확장 차원
    dropout: float = 0.0  # 과적합 실험 때 0.1 등으로 바꿔 비교할 예정

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd({self.n_embd})는 n_head({self.n_head})로 나누어떨어져야 한다"
            )

    @property
    def head_dim(self) -> int:
        """Head 하나의 차원 = Scaling에 쓰는 d_k (128 / 4 = 32)"""
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    """데이터와 학습 관련 설정"""

    data_path: Path = DATA_DIR / "input.txt"
    vocab_path: Path = DATA_DIR / "vocab.json"
    train_ratio: float = 0.9  # 앞 90%는 학습, 뒤 10%는 검증
    batch_size: int = 32  # 한 번에 처리하는 시퀀스 개수 (B)
    seed: int = 1337


def set_seed(seed: int) -> None:
    """재현 가능한 실험을 위해 모든 난수 생성기의 Seed를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU와 모든 GPU의 Seed를 함께 설정한다


def get_device() -> str:
    """GPU를 사용할 수 있으면 'cuda', 아니면 'cpu'를 반환한다."""
    return "cuda" if torch.cuda.is_available() else "cpu"
