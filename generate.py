"""텍스트 생성: 다음 문자를 하나씩 뽑아 입력 뒤에 붙이는 과정을 반복한다.

    uv run python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:"
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from config import GPTConfig, TrainConfig, get_device
from model import MiniGPT
from tokenizer import CharTokenizer

CHECKPOINT_VERSION = 1


def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """idx (B, T) 뒤에 max_new_tokens개의 토큰을 생성해 붙인다. 반환: (B, T + max_new_tokens)

    generator는 모델과 같은 device에 있어야 한다 (multinomial이 그 device에서 실행된다).
    """
    vocab_size = model.config.vocab_size
    if temperature <= 0:
        raise ValueError(f"temperature는 0보다 커야 한다: {temperature}")
    if top_k is not None and not 1 <= top_k <= vocab_size:
        raise ValueError(f"top_k는 1 ~ {vocab_size} 또는 None이어야 한다: {top_k}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens는 0 이상이어야 한다: {max_new_tokens}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 1. 모델은 block_size까지만 볼 수 있으므로 마지막 128개만 넣는다
                idx_cond = idx[:, -model.config.block_size :]
                logits, _ = model(idx_cond)  # (B, T, V)

                # 2. 마지막 위치의 Logits만 다음 토큰 예측에 사용한다: (B, V)
                logits = logits[:, -1, :] / temperature

                # 3. Top-k: 점수가 k번째보다 낮은 후보는 -inf로 제외한다
                if top_k is not None:
                    kth = torch.topk(logits, top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))

                # 4. 확률로 바꾼 뒤 하나를 뽑아 입력 뒤에 붙인다
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
                idx = torch.cat([idx, next_id], dim=1)
    finally:
        model.train(was_training)
    return idx


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
    top_k: int | None = 20,
    seed: int = 1337,
) -> str:
    """prompt로 시작하는 텍스트를 생성한다. 같은 seed와 device면 같은 결과가 나온다."""
    if not prompt:
        raise ValueError("prompt는 비어 있을 수 없다")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            f"토크나이저({tokenizer.vocab_size})와 모델({model.config.vocab_size})의 "
            "Vocabulary 크기가 다르다"
        )
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    # 생성 전용 난수: 학습 난수와 분리되어 있다
    generator = torch.Generator(device=device).manual_seed(seed)
    out = generate(model, idx, max_new_tokens, temperature, top_k, generator)
    return tokenizer.decode(out[0])


def load_checkpoint(path: str | Path) -> dict:
    """Checkpoint를 CPU로 읽는다.

    weights_only=True: 텐서와 기본 자료형만 읽으므로 임의 코드가 실행되지 않는다.
    map_location="cpu": RNG 상태 텐서는 CPU에 있어야 복원할 수 있다.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"지원하지 않는 Checkpoint 형식: {ckpt.get('format_version')}")
    return ckpt


def model_from_checkpoint(ckpt: dict, device: str) -> MiniGPT:
    model = MiniGPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="학습한 Mini GPT로 텍스트 생성")
    parser.add_argument("--checkpoint", default=str(TrainConfig().checkpoint_dir / "best.pt"))
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20, help="0이면 Top-k를 사용하지 않는다")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = get_device()
    ckpt = load_checkpoint(args.checkpoint)
    model = model_from_checkpoint(ckpt, device)
    tokenizer = CharTokenizer(ckpt["vocab"])  # 학습 때와 같은 Vocabulary

    print(f"checkpoint: {args.checkpoint} (step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})")
    print(
        f"device: {device} | prompt={args.prompt!r} | max_new_tokens={args.max_new_tokens} | "
        f"temperature={args.temperature} | top_k={args.top_k or None} | seed={args.seed}"
    )
    print("-" * 60)
    print(
        generate_text(
            model, tokenizer, args.prompt, args.max_new_tokens,
            args.temperature, args.top_k or None, args.seed,
        )
    )


if __name__ == "__main__":
    main()
