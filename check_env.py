"""개발 환경 확인: Python, PyTorch, CUDA, GPU

    uv run python check_env.py
"""

import platform
import sys

import torch


def main() -> None:
    print(f"Python : {sys.version.split()[0]}")
    print(f"OS     : {platform.system()} {platform.release()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("→ CPU로 실행된다. (토크나이저·데이터 단계는 CPU로도 충분하다)")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"PyTorch CUDA 버전: {torch.version.cuda}")
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1024**3:.2f} GiB")

    # GPU에서 실제 연산이 되는지 간단히 확인
    a = torch.randn(1024, 1024, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    print(f"GPU 행렬 곱 테스트: 성공 (결과 shape {tuple(b.shape)})")


if __name__ == "__main__":
    main()
