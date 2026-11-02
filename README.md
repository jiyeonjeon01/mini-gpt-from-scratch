# mini-gpt-from-scratch

PyTorch로 GPT 계열 Decoder-only Transformer를 처음부터 구현하고 학습하는 개인 프로젝트입니다.
사전 학습된 가중치는 사용하지 않습니다.

블로그 연재: LLM(SLM) 만들어보기

## 현재 진행 상황

- [x] 3편 #1: 프로젝트 구성, 문자 단위 토크나이저, 학습 데이터, get_batch
- [ ] 모델 구현 (Embedding, Attention, Transformer Block)
- [ ] 학습 루프
- [ ] 텍스트 생성

## 파일 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 모델·학습 설정, Seed 고정, Device 선택 |
| `tokenizer.py` | 문자 단위 토크나이저 (encode / decode / save / load) |
| `data.py` | 데이터 다운로드, Train/Validation 분리, 배치 생성 |
| `check_env.py` | Python, PyTorch, CUDA, GPU 확인 |
| `tests/` | pytest 테스트 |

## 실행 방법 (WSL2 Ubuntu)

```bash
# uv 설치 (처음 한 번)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 받기 (Windows 경로 /mnt/c 가 아니라 WSL 홈 아래에 둔다)
cd ~
git clone https://github.com/jiyeonjeon01/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch

# Python 3.12와 패키지 설치 (.venv 생성)
uv sync

# 환경 확인
uv run python check_env.py

# 데이터 준비 및 배치 확인
uv run python data.py

# 테스트
uv run pytest -v
```

Linux에서는 PyPI의 PyTorch가 CUDA 포함 버전으로 설치되고,
Windows에서는 CPU 버전으로 설치됩니다. 토크나이저와 데이터 단계는 CPU로도 충분합니다.

처음 `uv sync`를 실행하면 `uv.lock`이 생성됩니다. 같은 버전으로 재현할 수 있도록 함께 커밋합니다.

## 모델 설정 (v1)

| 설정 | 값 |
|---|---|
| block_size | 128 |
| n_embd | 128 |
| n_head | 4 |
| n_layer | 4 |
| ffn_dim | 512 |


## License

Copyright (c) 2026 Jiyeon Jeon. All rights reserved.
자세한 내용은 [LICENSE](LICENSE)를 참고하세요.
