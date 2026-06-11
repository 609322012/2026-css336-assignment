"""Section 2: byte-level BPE tokenizer training."""

import os
import re

import regex

PRETOKENIZE_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)|\s+"
)


def BEP(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    escaped = [re.escape(tok) for tok in special_tokens]
    pattern = re.compile(f"({'|'.join(escaped)})")
    segments = re.split(pattern, text)

    tokens: list[str] = []
    for segment in segments:
        if segment not in special_tokens:
            tokens.extend(regex.findall(PRETOKENIZE_PATTERN, segment))

    word_freq: dict[tuple[bytes, ...], int] = {}
    for token in tokens:
        word = tuple(bytes([b]) for b in token.encode("utf-8"))
        word_freq[word] = word_freq.get(word, 0) + 1

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        counts: dict[tuple[bytes, bytes], int] = {}
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                counts[pair] = counts.get(pair, 0) + freq

        if not counts:
            break

        best_pair = max(counts, key=lambda p: (counts[p], p))
        new_token = best_pair[0] + best_pair[1]

        word_freq_new: dict[tuple[bytes, ...], int] = {}
        for word, freq in word_freq.items():
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            merged = tuple(new_word)
            word_freq_new[merged] = word_freq_new.get(merged, 0) + freq

        word_freq = word_freq_new
        vocab[len(vocab)] = new_token
        merges.append(best_pair)

    return vocab, merges
