# File: project_chimera/l2_data/unigram_tokenizer.py
import numpy as np
import heapq
from collections import Counter

class UnigramTokenizer:
    """
    Implements a Unigram Language Model Tokenizer.
    This version includes a fix for the Viterbi algorithm pathfinding.
    """
    def __init__(self, vocab_size=300, max_subword_len=16):
        self.vocab_size = vocab_size
        self.max_subword_len = max_subword_len
        self.vocab = {}
        self.scores = {}

    def train(self, corpus, num_iterations=5):
        print("[L2 ASTF] Training Unigram tokenizer (MDL-optimized)...")
        initial_vocab = {chr(i): i for i in range(256)}
        
        all_substrings = Counter()
        for text in corpus:
            for i in range(len(text)):
                for j in range(i + 1, min(i + 1 + self.max_subword_len, len(text) + 1)):
                    all_substrings[text[i:j]] += 1
        
        self.vocab = initial_vocab
        sorted_substrings = sorted(all_substrings.items(), key=lambda x: x[1], reverse=True)
        for sub, count in sorted_substrings:
            if len(self.vocab) >= self.vocab_size * 1.5:
                break
            if sub not in self.vocab:
                self.vocab[sub] = len(self.vocab)
        
        self.scores = {token: np.log(1 / len(self.vocab)) for token in self.vocab.keys()}
        for i in range(num_iterations):
            _, token_counts = self._viterbi_tokenize_corpus(corpus)
            total_sum = sum(token_counts.values())
            if total_sum == 0: continue

            self.scores = {token: np.log(count / total_sum) for token, count in token_counts.items()}
            
            if len(self.vocab) > self.vocab_size:
                num_to_prune = len(self.vocab) - self.vocab_size
                prunable_tokens = {k: v for k, v in token_counts.items() if len(k) > 1}
                tokens_to_prune = sorted(prunable_tokens.items(), key=lambda x: x[1])[:num_to_prune]
                for token, _ in tokens_to_prune:
                    if token in self.vocab:
                        del self.vocab[token]
        
        self.vocab = {token: i for i, token in enumerate(self.vocab.keys())}
        final_counts = self._viterbi_tokenize_corpus(corpus)[1]
        total_sum = sum(final_counts.values()) if sum(final_counts.values()) > 0 else 1
        self.scores = {token: np.log(count / total_sum) if count > 0 else -float('inf') for token, count in final_counts.items()}
        
        self.id_to_token = {i: token for token, i in self.vocab.items()}
        print(f"[L2 ASTF] Unigram training complete. Final vocab size: {len(self.vocab)}")

    def _viterbi_tokenize_corpus(self, corpus):
        tokenized_corpus, full_counts = [], Counter()
        for text in corpus:
            result = self.encode(text, return_counts=True)
            if result:
                tokens, counts = result
                tokenized_corpus.append(tokens)
                full_counts.update(counts)
        return tokenized_corpus, full_counts

    def encode(self, text, return_counts=False):
        N = len(text)
        if N == 0: return ([], Counter()) if return_counts else []

        dp = [float('inf')] * (N + 1)
        backpointers = [None] * (N + 1)
        dp[0] = 0

        for i in range(1, N + 1):
            for j in range(max(0, i - self.max_subword_len), i):
                subword = text[j:i]
                if subword in self.scores:
                    score = -self.scores.get(subword, float('inf'))
                    if dp[j] != float('inf') and dp[j] + score < dp[i]:
                        dp[i] = dp[j] + score
                        backpointers[i] = j
        
        if dp[N] == float('inf'):
            tokens = [c for c in list(text) if c in self.vocab]
        else:
            tokens = []
            i = N
            while i > 0:
                j = backpointers[i]
                if j is None:
                    tokens = [c for c in list(text) if c in self.vocab]
                    break
                tokens.insert(0, text[j:i])
                i = j
        
        valid_tokens = [t for t in tokens if t in self.vocab]
        
        if return_counts:
            return [self.vocab[t] for t in valid_tokens], Counter(valid_tokens)
        return [self.vocab[t] for t in valid_tokens]

    def decode(self, ids):
        return "".join(self.id_to_token.get(i, "?") for i in ids)
