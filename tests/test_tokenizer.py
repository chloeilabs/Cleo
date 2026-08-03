from __future__ import annotations

import json

from cleo1.tokenizer import ByteBPETokenizer


CORPUS = (
    b"Once upon a time there was a small cat.\n\n"
    b"The small cat liked the warm sun and the green garden.\n\n"
    b"Unicode examples are encoded as bytes: \xf0\x9f\x8c\x9f \xe2\x98\x83.\n\n"
) * 20


def test_byte_round_trip_including_unicode_and_all_byte_values(tmp_path):
    tokenizer = ByteBPETokenizer.train(CORPUS, vocab_size=280, metadata={"test": True})
    payload = bytes(range(256)) + " hello 🌟\n".encode()
    encoded = tokenizer.encode_bytes(payload, bos=True, eos=True)
    assert tokenizer.decode_bytes(encoded) == payload
    assert tokenizer.bos_id in encoded
    assert tokenizer.eos_id in encoded
    assert all(0 <= token < tokenizer.vocab_size for token in encoded)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.encode_bytes(payload, bos=True, eos=True) == encoded
    assert loaded.decode_bytes(encoded) == payload


def test_training_is_deterministic_and_uses_lexicographic_tie_break():
    first = ByteBPETokenizer.train(CORPUS, vocab_size=280)
    second = ByteBPETokenizer.train(CORPUS, vocab_size=280)
    assert first.merges == second.merges
    assert first.to_dict() == second.to_dict()
