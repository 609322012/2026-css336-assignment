"""Section 2: byte-level BPE tokenizer training."""

import os
import re
from secrets import token_bytes

import regex

PRETOKENIZE_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)|\s+"
)

import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def BEP(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_processes = 8 ##process workers nums

    word_freq: dict[tuple[bytes, ...], int] = {}
    escaped = [re.escape(tok) for tok in special_tokens]
    pattern = re.compile(f"({'|'.join(escaped)})")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):

            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            segments = re.split(pattern, chunk)

            for segment in segments:
                if segment not in special_tokens:
                    for token in regex.findall(PRETOKENIZE_PATTERN, segment):
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

# def get_tokenizer(
#     vocab: dict[int, bytes],
#     merges: list[tuple[bytes, bytes]],
#     special_tokens: list[str] | None = None,
# ):
#     id = next((k for k, v in vocab.items() if v == bytes), None)


class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = []

        vocab_new = {}
        for token_id, token_bytes in vocab.items():
            vocab_new[token_bytes] = token_id
        self.vocab_new = vocab_new

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        else:
            result = []
            for token in regex.findall(PRETOKENIZE_PATTERN, text):
                if token in self.special_tokens:
                    result.append(self.vocab_new[token])
                else:
                    token_bytes = token.encode("utf-8")
                    id = self.vocab_new[token_bytes]
                    result.append(id)

            return result
            
    def decode(self, ids: list[int]) -> str:
        pieces = []
        for toekn_id in ids:
                pieces.append(self.vocab[toekn_id])
        result = b''.join(pieces)
        result = result.decode("utf-8")
        return result

    def bpe_merge(self, token: bytes) -> bytes:
        bytes = token.encode("utf-8")
        word = tuple(bytes([b]) for b in bytes.encode("utf-8"))
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair in self.merges:
                return self.merges[pair]
        return word