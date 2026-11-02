"""문자 단위 토크나이저 (2편 예제 코드를 클래스로 정리한 것)"""

import json
from pathlib import Path


class CharTokenizer:
    """문자 하나를 토큰 하나로 취급하는 토크나이저

    - stoi: String to Integer (문자 → 토큰 ID)
    - itos: Integer to String (토큰 ID → 문자)
    """

    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("Vocabulary에 중복된 문자가 있다")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """텍스트에 등장한 문자들을 중복 제거 후 정렬해 Vocabulary를 만든다."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """문자열 → 토큰 ID 목록"""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise ValueError(f"Vocabulary에 없는 문자: {e.args[0]!r}") from None

    def decode(self, ids) -> str:
        """토큰 ID 목록 → 문자열 (list와 1차원 Tensor 모두 받을 수 있다)"""
        try:
            return "".join(self.itos[int(i)] for i in ids)
        except KeyError as e:
            raise ValueError(
                f"Vocabulary에 없는 토큰 ID: {e.args[0]} "
                f"(유효 범위 0 ~ {self.vocab_size - 1})"
            ) from None

    def save(self, path: str | Path) -> None:
        """Vocabulary를 JSON으로 저장한다. 학습과 생성에서 같은 ID 체계를 쓰기 위해 필요하다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])
