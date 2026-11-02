"""CharTokenizer 테스트 (torch 없이 실행 가능)"""

import pytest

from tokenizer import CharTokenizer


def test_blog_example():
    """2편 예제와 같은 결과가 나와야 한다."""
    tok = CharTokenizer.from_text("I love AI")
    assert tok.chars == [" ", "A", "I", "e", "l", "o", "v"]
    assert tok.vocab_size == 7
    assert tok.encode("I love AI") == [2, 0, 4, 5, 6, 3, 0, 1, 2]


def test_roundtrip():
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_raises():
    """2편: 학습 때 없던 문자는 처리할 수 없다."""
    tok = CharTokenizer.from_text("I love AI")
    with pytest.raises(ValueError, match="Vocabulary에 없는 문자"):
        tok.encode("I love ML")


def test_duplicate_chars_rejected():
    with pytest.raises(ValueError):
        CharTokenizer(["a", "b", "a"])


def test_save_and_load(tmp_path):
    tok = CharTokenizer.from_text("Hello, World!\n")
    path = tmp_path / "vocab.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.chars == tok.chars
    assert loaded.encode("Hello") == tok.encode("Hello")
