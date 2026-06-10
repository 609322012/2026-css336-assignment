from ast import Tuple
import os
import regex
from typing import Any, BinaryIO


def BEP(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    rule = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    with open(input_path,"r",encoding="utf-8") as f:
      text = f.read() 
  
    tokens = regex.findall(rule, text)

    word_freq = {}

    for token in tokens:
        word = tuple(bytes([b]) for b in token.encode("utf-8"))
        word_freq[word] = word_freq.get(word , 0) + 1

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    counts_max = 0
    merges = []

    while(len(vocab) < vocab_size):
        counts = {}
        for word in word_freq:
            for i in range(len(word) -  1):
                pair = (word[i],word[i+1])
                counts[pair] = counts.get(pair,0) + word_freq[word];
        
        if not counts:
            break;
        else:
            counts_max = max(counts, key= lambda p:(counts[p],p))
            new_token = counts_max[0] + counts_max[1];

        word_freq_new = {} 
        result = []
        for word in word_freq:
            new_word = []
            i = 0
            while(i < len(word)):
                if i < len(word) - 1 and word[i] == counts_max[0] and word[i+1] == counts_max[1]:
                    new_word.append(new_token)
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            
            tmp = tuple(new_word)
            word_freq_new[tmp] = word_freq_new.get(tmp,0) + word_freq[word]
        word_freq = word_freq_new
            # result.append(tmp)

        # word_freq = result
        vocab[len(vocab)] = new_token
        merges.append(counts_max)
  
    return vocab, merges