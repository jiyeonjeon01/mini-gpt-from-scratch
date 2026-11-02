"""Mini GPT 모델: Decoder-only Transformer (Pre-LN, Causal Self-Attention)

실행하면 실제 데이터 배치 하나를 모델에 넣어 Shape과 초기 Loss를 출력한다.
    uv run python model.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, TrainConfig, get_device, set_seed
from data import download_data, get_batch, load_text, split_data
from tokenizer import CharTokenizer


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention

    각 위치는 자기 자신과 그 이전 위치의 토큰만 참고할 수 있다.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd

        # Q, K, V를 Linear 하나로 한 번에 계산한다: 128 → 384 (= 128 × 3)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Head들을 합친 뒤 적용하는 출력 Projection: 128 → 128
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 아래 삼각형(대각선 포함)만 True인 Causal Mask: (1, 1, block_size, block_size)
        # buffer로 등록하면 model.to(device) 때 함께 이동하고, 학습 파라미터에는 포함되지 않는다.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (B, T, 128)

        # 1. Q, K, V 계산 후 분리: 각각 (B, T, 128)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Head 분리: (B, T, 128) → (B, T, 4, 32) → (B, 4, T, 32)
        #    view(B, 4, T, 32)로 바로 바꾸면 Shape은 같지만 위치가 섞이므로 transpose가 필요하다.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Attention Score: (B, 4, T, 32) @ (B, 4, 32, T) → (B, 4, T, T), 1/√32로 Scaling
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Causal Mask: 미래 위치(위 삼각형)를 -inf로 바꿔 Softmax 결과가 0이 되게 한다.
        #    대각선은 항상 열려 있으므로 한 행 전체가 -inf가 되는 일은 없다.
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 5. Value 가중합: (B, 4, T, T) @ (B, 4, T, 32) → (B, 4, T, 32)
        y = att @ v

        # 6. Head 병합: (B, 4, T, 32) → (B, T, 4, 32) → (B, T, 128)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """위치별 FFN: 128 → 512 → GELU → 128"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.ffn_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(config.ffn_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block

        H = X + Attention(LN1(X))
        Y = H + FFN(LN2(H))
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """토큰 ID (B, T)를 받아 각 위치의 다음 토큰 Logits (B, T, vocab_size)를 출력한다."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("config.vocab_size를 먼저 설정해야 한다 (tokenizer.vocab_size)")
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # Learned Absolute
        self.drop = nn.Dropout(config.dropout)
        # Block마다 독립된 인스턴스를 만든다 (같은 객체를 반복하면 가중치가 공유된다)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Token Embedding과 가중치를 공유하지 않는다
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 방식: 가중치 N(0, 0.02), Linear bias 0, LayerNorm weight 1 · bias 0"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     (B, T) 입력 토큰 ID (torch.long)
        targets: (B, T) 정답 토큰 ID (선택)

        반환: logits (B, T, vocab_size), loss (targets가 없으면 None)
        """
        if idx.ndim != 2:
            raise ValueError(f"idx는 (B, T) 형태여야 한다: {tuple(idx.shape)}")
        if idx.dtype != torch.long:
            raise ValueError(f"idx의 dtype은 torch.long이어야 한다: {idx.dtype}")
        B, T = idx.shape
        if not 1 <= T <= self.config.block_size:
            raise ValueError(f"입력 길이 T={T}는 1 ~ block_size({self.config.block_size})여야 한다")
        if targets is not None and (targets.shape != idx.shape or targets.dtype != torch.long):
            raise ValueError(
                f"targets는 idx와 같은 Shape의 torch.long이어야 한다: "
                f"{tuple(targets.shape)} {targets.dtype}"
            )

        # 1. Token Embedding (B, T, 128) + Position Embedding (T, 128) → (B, T, 128)
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # 2. Transformer Block × n_layer
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm → LM Head: (B, T, 128) → (B, T, vocab_size)
        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            return logits, None

        # cross_entropy는 클래스 차원을 두 번째 축으로 보므로 (B*T, V)와 (B*T,)로 편다
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def main() -> None:
    train_cfg = TrainConfig()
    set_seed(train_cfg.seed)
    device = get_device()

    # 1. 데이터와 토크나이저 준비 (data.py와 같은 흐름)
    download_data(train_cfg.data_path)
    text = load_text(train_cfg.data_path)
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, _ = split_data(data, train_cfg.train_ratio)

    # 2. 모델 생성 (vocab_size는 실제 데이터에서 결정)
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = MiniGPT(config).to(device)
    model.eval()

    print(f"device: {device}")
    print(
        f"설정: vocab_size={config.vocab_size}, block_size={config.block_size}, "
        f"n_embd={config.n_embd}, n_head={config.n_head} (head_dim={config.head_dim}), "
        f"n_layer={config.n_layer}, ffn_dim={config.ffn_dim}, dropout={config.dropout}"
    )

    # 3. 파라미터 수와 구성
    def count(*modules: nn.Module) -> int:
        return sum(p.numel() for m in modules for p in m.parameters())

    print(f"파라미터 수: {model.num_parameters():,}")
    print(f"  - 임베딩 (token + position)  : {count(model.tok_emb, model.pos_emb):,}")
    print(f"  - Transformer Block × {config.n_layer}     : {count(model.blocks):,}")
    print(f"  - Final LayerNorm + LM Head : {count(model.ln_f, model.lm_head):,}")

    # 4. 실제 배치 하나로 Forward
    x, y = get_batch(train_data, train_cfg.batch_size, config.block_size, device)
    with torch.no_grad():
        logits, loss = model(x, y)

    print(f"입력 x : {tuple(x.shape)} {x.dtype}")
    print(f"logits : {tuple(logits.shape)}")
    print(f"초기 loss: {loss.item():.4f}  (참고: 균등분포 기준 ln({config.vocab_size}) = "
          f"{math.log(config.vocab_size):.4f})")


if __name__ == "__main__":
    main()
