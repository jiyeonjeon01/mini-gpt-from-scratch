# mini-gpt-from-scratch

PyTorch로 GPT 계열 Decoder-only Transformer를 처음부터 구현하고, 무작위 가중치부터 학습하는 교육용 개인 프로젝트입니다.
사전 학습된 가중치는 사용하지 않습니다.

블로그 연재: LLM(SLM) 만들어보기

## 현재 진행 상황

- [x] 3편 #1: 프로젝트 구성, 문자 단위 토크나이저, 학습 데이터 준비, Train/Validation 분리, `get_batch`
- [x] 3편 #2: 모델 구현 (Token/Position Embedding, Causal Self-Attention, Pre-LN Transformer Block, LM Head, Loss)
- [ ] 학습 루프 (AdamW, Train/Validation Loss)
- [ ] 텍스트 생성

## 모델 설정 (v1, 블로그 '6. 모델 설계' 기준)

| 설정 | 값 | 설명 |
|---|---|---|
| block_size | 128 | 한 번에 참고하는 최대 토큰 수 (T) |
| n_embd | 128 | 임베딩 차원 (C) |
| n_head | 4 | Attention Head 수 (head_dim = 32) |
| n_layer | 4 | Transformer Block 수 |
| ffn_dim | 512 | FFN 내부 차원 |

- 구조: Character-level Tokenizer, Learned Positional Embedding, Multi-Head Causal Self-Attention,
  Pre-LN Transformer, GELU FFN, AdamW (AdamW는 학습 단계에서 사용 예정)
- 모델은 `model.py`에 구현되어 있습니다. 입력 `(B, T)` 토큰 ID → 출력 `(B, T, vocab_size)` Logits
- 파라미터 수: **826,433개** (vocab_size=65, Linear bias 포함, Token Embedding과 LM Head 가중치 비공유)
  - 계산식: `809,728 + 257 × vocab_size` (vocab_size=100으로 가정하면 835,428)
- 초기화: Linear·Embedding 가중치 N(0, 0.02), Linear bias 0, LayerNorm weight 1 · bias 0
- Dropout 위치: Embedding 합 뒤, Attention Softmax 뒤, Attention 출력 Projection 뒤, FFN 출력 뒤 (현재 모두 0.0)

```text
idx (B, T)
 → Token Embedding (B, T, 128) + Position Embedding (T, 128)
 → [LN → Causal Multi-Head Attention → Residual → LN → FFN(128→512→128) → Residual] × 4
 → Final LayerNorm → LM Head (128 → vocab_size)
 → logits (B, T, vocab_size), targets가 있으면 Cross-Entropy loss
```

## 파일 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 모델·학습 설정(`GPTConfig`, `TrainConfig`), 프로젝트 루트 기준 데이터 경로, Seed 고정, Device 선택 |
| `tokenizer.py` | 문자 단위 토크나이저 (encode / decode / save / load) |
| `data.py` | 데이터 다운로드, Train/Validation 분리, 배치 생성. 직접 실행하면 전체 과정을 출력 |
| `model.py` | Mini GPT 모델 (`CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `MiniGPT`). 직접 실행하면 파라미터 수, Shape, 초기 loss를 출력 |
| `check_env.py` | Python, PyTorch, CUDA, GPU 확인 |
| `tests/` | pytest 테스트 (토크나이저, 데이터 분리·배치, 다운로드, 모델) |
| `pyproject.toml`, `uv.lock` | 의존성 선언과 재현용 고정 버전 |

## 설치와 실행 (WSL2 Ubuntu, Python 3.12, uv)

```bash
# uv 설치 (처음 한 번, 사용자 홈 ~/.local/bin 에 설치됨)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 받기 (Windows 경로 /mnt/c 가 아니라 WSL 홈 아래 ~/dev 에 둔다)
mkdir -p ~/dev && cd ~/dev
git clone https://github.com/jiyeonjeon01/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch

# uv.lock 에 고정된 버전 그대로 .venv 생성 및 패키지 설치
uv sync --locked

# 환경 확인 (PyTorch, CUDA, GPU)
uv run python check_env.py

# 데이터 준비 및 배치 확인
uv run python data.py

# 모델 구조 확인 (파라미터 수, logits Shape, 초기 loss)
uv run python model.py

# 테스트
uv run pytest -q
```

- Python 버전은 `.python-version`과 `pyproject.toml`(`>=3.12,<3.13`)로 3.12에 고정합니다.
- Linux에서는 PyPI의 PyTorch가 CUDA 포함 버전으로 설치되므로 WSL 안에 CUDA Toolkit을 따로 설치할 필요가 없습니다.
  (GPU 드라이버는 Windows 쪽 NVIDIA 드라이버를 사용합니다.)
- GPU가 없으면 자동으로 CPU를 사용합니다. 토크나이저와 데이터 단계는 CPU로도 충분합니다.
- 데이터 경로는 프로젝트 루트 기준으로 고정되어 있어, 어느 디렉터리에서 실행해도 같은 `data/`를 사용합니다.

## 학습 데이터

- 데이터: Tiny Shakespeare (셰익스피어 희곡 일부를 이어 붙인 영어 텍스트, 1,115,394 bytes)
- 출처: Andrej Karpathy의 [char-rnn](https://github.com/karpathy/char-rnn) 저장소
  `data/tinyshakespeare/input.txt`
- 라이선스: 셰익스피어 원작은 저작권 보호 기간이 지난 작품이지만, 위 저장소에는
  별도의 라이선스 파일이 확인되지 않습니다. 이 저장소는 데이터를 포함하지 않고,
  `data.py` 실행 시 원본 URL에서 내려받기만 합니다.
- 저장 위치: 프로젝트 루트의 `data/input.txt`, `data/vocab.json` (`.gitignore`로 커밋 제외)
- 다운로드 방식: `data/input.txt.part`에 먼저 받고, 받은 크기가 서버가 알려준 크기(Content-Length)와
  같을 때만 `input.txt`로 교체합니다. 중간에 실패하면 `input.txt`가 만들어지지 않으므로
  다시 실행하면 처음부터 내려받습니다. 이미 `input.txt`가 있으면 다시 받지 않습니다.

## 문자 단위 토크나이저

- 텍스트에 등장한 문자를 중복 제거 후 정렬하여 Vocabulary를 만들고, 문자 하나에 토큰 ID 하나를 붙입니다.
  Tiny Shakespeare 기준 Vocabulary Size는 65입니다.
- `encode`: 문자열 → 토큰 ID 목록. Vocabulary에 없는 문자는 `ValueError`
- `decode`: 토큰 ID 목록(list 또는 1차원 Tensor) → 문자열. 범위 밖 ID는 `ValueError`
- `save` / `load`: Vocabulary를 JSON으로 저장·복원하여 학습과 생성에서 같은 ID 체계를 사용합니다.

## Train/Validation 분리와 Vocabulary 생성 기준

- **Vocabulary(문자 → ID 표)는 전체 텍스트에서 만듭니다.** 검증 구간에만 나오는 문자가
  있어도 encode 오류 없이 처리하기 위해서입니다. 이때 검증 텍스트에서 가져오는 정보는
  "어떤 문자가 존재하는가"뿐입니다. (Tiny Shakespeare는 검증 구간에만 있는 문자가 없습니다.)
- **학습·검증 텍스트는 분리합니다.** 토큰 열의 앞 90%는 학습, 뒤 10%는 검증으로 섞지 않고
  연속해서 나누며, 배치(x, y)는 각 구간 안에서만 뽑습니다. 검증 문장은 학습에 쓰이지 않습니다.
- 다음 토큰 예측을 위해 정답 `y`는 입력 `x`를 한 칸 뒤로 민 값입니다.
  `x = data[i : i+T]`, `y = data[i+1 : i+T+1]`, Shape은 둘 다 `(B, T)`입니다.

## 테스트

```bash
uv run pytest -q     # 전체
uv run pytest -v     # 테스트별 결과
```

다운로드 테스트는 네트워크 없이 가짜 응답으로 실행됩니다.
모델 테스트에는 미래 토큰을 바꿔도 과거 위치의 Logits가 변하지 않는지, 미래 위치로 Gradient가 흐르지 않는지
확인하는 Causal Mask 테스트가 포함되어 있습니다 (`uv run pytest tests/test_model.py -v -k causal`).
CUDA 테스트는 GPU를 사용할 수 있을 때만 실행됩니다.

## 다음 단계

- `train.py`: AdamW 학습 루프, Train/Validation Loss 측정, 체크포인트 저장
- 텍스트 생성 (`generate`)

## License

Copyright (c) 2026 Jiyeon Jeon. All rights reserved.
자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
