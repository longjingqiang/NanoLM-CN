"""
NanoLM 中文分词器
支持字符级、BPE 两种模式，专为中文文本优化
"""

import os
import re
import json
import collections
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path


# ─── 特殊 Token ───────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "<pad>":  0,
    "<unk>":  1,
    "<bos>":  2,
    "<eos>":  3,
    "<sep>":  4,
    "<mask>": 5,
}


# ─── 中文预处理 ───────────────────────────────────────────────────────────────

def normalize_chinese_text(text: str) -> str:
    """标准化中文文本"""
    # 全角转半角
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角字母/标点
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:           # 全角空格
            result.append(" ")
        else:
            result.append(ch)
    text = "".join(result)
    # 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_chinese_char(ch: str) -> bool:
    """判断是否为中文字符"""
    cp = ord(ch)
    return (
        (0x4E00 <= cp <= 0x9FFF)   or   # CJK 基本区
        (0x3400 <= cp <= 0x4DBF)   or   # CJK 扩展 A
        (0x20000 <= cp <= 0x2A6DF) or   # CJK 扩展 B
        (0x2A700 <= cp <= 0x2B73F) or
        (0xF900 <= cp <= 0xFAFF)   or   # CJK 兼容区
        (0x2F800 <= cp <= 0x2FA1F)
    )


def tokenize_chinese(text: str) -> List[str]:
    """
    中文分词预处理：
    - 中文字符每个单独成词
    - 英文/数字保留完整单词
    - 标点符号单独处理
    """
    tokens = []
    current_word = []

    for ch in text:
        if is_chinese_char(ch):
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            tokens.append(ch)
        elif ch.isalnum() or ch in "-_'":
            current_word.append(ch)
        else:
            if current_word:
                tokens.append("".join(current_word))
                current_word = []
            if ch.strip():
                tokens.append(ch)
            elif current_word or tokens:
                tokens.append(" ")

    if current_word:
        tokens.append("".join(current_word))

    return [t for t in tokens if t]


# ─── 字符级分词器 ─────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    字符级分词器（最简单，适合小数据集）
    中文字符每个单独编码，英文按字符编码
    """

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for tok, idx in SPECIAL_TOKENS.items():
            self.token2id[tok] = idx
            self.id2token[idx] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int: return SPECIAL_TOKENS["<pad>"]
    @property
    def unk_id(self) -> int: return SPECIAL_TOKENS["<unk>"]
    @property
    def bos_id(self) -> int: return SPECIAL_TOKENS["<bos>"]
    @property
    def eos_id(self) -> int: return SPECIAL_TOKENS["<eos>"]

    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab: int = 8000):
        """从文本语料构建词表"""
        print(f"构建词表中 (min_freq={min_freq}, max_vocab={max_vocab})...")
        freq: Dict[str, int] = collections.Counter()

        for text in texts:
            text = normalize_chinese_text(text)
            for ch in text:
                freq[ch] += 1

        # 按频率排序
        sorted_chars = sorted(freq.items(), key=lambda x: -x[1])
        n_special = len(SPECIAL_TOKENS)
        max_new = max_vocab - n_special

        added = 0
        for char, count in sorted_chars:
            if count < min_freq or added >= max_new:
                break
            if char not in self.token2id:
                idx = len(self.token2id)
                self.token2id[char] = idx
                self.id2token[idx] = char
                added += 1

        print(f"词表构建完成: {self.vocab_size} 个 token "
              f"(特殊={n_special}, 字符={added})")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """编码文本为 token ids"""
        text = normalize_chinese_text(text)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self.token2id.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """解码 token ids 为文本"""
        special_ids = set(SPECIAL_TOKENS.values())
        chars = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            chars.append(self.id2token.get(i, "?"))
        return "".join(chars)

    def encode_batch(self, texts: List[str], max_len: int, 
                     pad: bool = True) -> Tuple[List[List[int]], List[int]]:
        """批量编码，返回 (ids, lengths)"""
        encoded = [self.encode(t)[:max_len] for t in texts]
        lengths = [len(e) for e in encoded]
        if pad:
            for ids in encoded:
                ids.extend([self.pad_id] * (max_len - len(ids)))
        return encoded, lengths

    def save(self, path: str):
        """保存分词器"""
        os.makedirs(path, exist_ok=True)
        data = {
            "type": "char",
            "token2id": self.token2id,
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"分词器已保存至 {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """加载分词器"""
        tokenizer = cls()
        with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.token2id = {k: int(v) for k, v in data["token2id"].items()}
        tokenizer.id2token = {int(v): k for k, v in data["token2id"].items()}
        print(f"分词器已加载: vocab_size={tokenizer.vocab_size}")
        return tokenizer


# ─── BPE 分词器 ───────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    字节对编码 (BPE) 分词器
    比字符级更高效，可以处理未登录词
    适合较大词表 (8000~16000)
    """

    def __init__(self):
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for tok, idx in SPECIAL_TOKENS.items():
            self.token2id[tok] = idx
            self.id2token[idx] = tok

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int: return SPECIAL_TOKENS["<pad>"]
    @property
    def unk_id(self) -> int: return SPECIAL_TOKENS["<unk>"]
    @property
    def bos_id(self) -> int: return SPECIAL_TOKENS["<bos>"]
    @property
    def eos_id(self) -> int: return SPECIAL_TOKENS["<eos>"]

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def train(self, texts: List[str], vocab_size: int = 8000,
              min_freq: int = 2, verbose: bool = True):
        """训练 BPE 词表"""
        target_merges = vocab_size - len(SPECIAL_TOKENS)
        if verbose:
            print(f"开始 BPE 训练 (目标词表: {vocab_size}, 需合并: {target_merges} 次)...")

        # 1. 字符级初始化
        word_freq: Dict[str, int] = collections.Counter()
        for text in texts:
            text = normalize_chinese_text(text)
            words = tokenize_chinese(text)
            for word in words:
                # 中文字符单字化；英文词添加 </w> 结尾标记
                if all(is_chinese_char(c) for c in word):
                    for ch in word:
                        word_freq[ch] += 1
                else:
                    word_freq[word + "</w>"] += 1

        # 构建初始词表（字符级）
        char_freq: Dict[str, int] = collections.Counter()
        vocab_words: Dict[str, List[str]] = {}  # word → [chars]

        for word, freq in word_freq.items():
            if freq < min_freq:
                continue
            chars = list(word)
            vocab_words[word] = chars
            for ch in chars:
                char_freq[ch] += freq

        # 若字符数超过目标词表，只保留频率最高的字符（其余变 <unk>）
        n_special = len(SPECIAL_TOKENS)
        max_chars = vocab_size - n_special - 200  # 留 200 个槽位给 BPE merge
        if len(char_freq) > max_chars:
            # 保留高频字符
            top_chars = sorted(char_freq.items(), key=lambda x: -x[1])[:max_chars]
            allowed_chars = {ch for ch, _ in top_chars}
            char_freq = {ch: f for ch, f in char_freq.items() if ch in allowed_chars}
            # 重建 vocab_words，未知字符合并为 <unk>
            for word in list(vocab_words.keys()):
                new_syms = [ch if ch in allowed_chars else "<unk>" for ch in vocab_words[word]]
                vocab_words[word] = new_syms
            # 重建词表（清空再重加）
            self.token2id = {}
            self.id2token = {}
            self._init_special_tokens()
            if verbose:
                print(f"  字符数超限 ({len(char_freq)+len(top_chars[max_chars:])}) → "
                      f"保留高频 {max_chars} 个字符")

        # 添加基础字符到词表（覆盖 train() 之前的重复调用）
        for ch in sorted(char_freq.keys()):
            self._add_token(ch)

        n_merges_needed = vocab_size - len(self.token2id)
        if verbose:
            print(f"基础字符数: {len(char_freq)}, 需要 BPE 合并: {n_merges_needed} 次, "
                  f"训练词数: {len(vocab_words)}")

        # 2. BPE 合并循环
        for merge_idx in range(n_merges_needed):
            # 统计相邻对频率
            pair_freq: Dict[Tuple[str, str], int] = collections.Counter()
            for word, freq in word_freq.items():
                if freq < min_freq or word not in vocab_words:
                    continue
                symbols = vocab_words[word]
                for i in range(len(symbols) - 1):
                    pair_freq[(symbols[i], symbols[i+1])] += freq

            if not pair_freq:
                break

            # 选择最高频对
            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] < min_freq:
                break

            # 合并
            new_token = "".join(best_pair)
            self.merges[best_pair] = new_token
            self._add_token(new_token)

            # 更新 vocab_words
            for word in list(vocab_words.keys()):
                syms = vocab_words[word]
                new_syms = []
                i = 0
                while i < len(syms):
                    if (i < len(syms) - 1 and
                            syms[i] == best_pair[0] and syms[i+1] == best_pair[1]):
                        new_syms.append(new_token)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                vocab_words[word] = new_syms

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"  已合并 {merge_idx+1}/{target_merges} 次，"
                      f"当前词表: {self.vocab_size}")

        if verbose:
            print(f"BPE 训练完成: {self.vocab_size} 个 token，{len(self.merges)} 条合并规则")

    def _apply_bpe(self, word: str) -> List[str]:
        """对一个词应用 BPE 合并"""
        if all(is_chinese_char(c) for c in word):
            return list(word)

        symbols = list(word + "</w>") if not word.endswith("</w>") else list(word)

        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            # 找到优先级最高（最先被合并）的对
            mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]
            if not mergeable:
                break
            best_pair, merged = min(
                mergeable,
                key=lambda x: list(self.merges.keys()).index(x[0])
            )
            new_syms = []
            i = 0
            while i < len(symbols):
                if (i < len(symbols)-1 and
                        symbols[i] == best_pair[0] and symbols[i+1] == best_pair[1]):
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms

        return symbols

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """编码文本"""
        text = normalize_chinese_text(text)
        words = tokenize_chinese(text)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for word in words:
            for subword in self._apply_bpe(word):
                ids.append(self.token2id.get(subword, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """解码 token ids"""
        special_ids = set(SPECIAL_TOKENS.values())
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tok = self.id2token.get(i, "?")
            tokens.append(tok)
        # 去除 BPE 结尾标记
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def save(self, path: str):
        """保存分词器"""
        os.makedirs(path, exist_ok=True)
        data = {
            "type": "bpe",
            "token2id": self.token2id,
            "merges": [[a, b] for (a, b) in self.merges.keys()],
            "special_tokens": SPECIAL_TOKENS,
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"BPE 分词器已保存至 {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """加载分词器"""
        tokenizer = cls()
        with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)
        tokenizer.token2id = {k: int(v) for k, v in data["token2id"].items()}
        tokenizer.id2token = {int(v): k for k, v in data["token2id"].items()}
        tokenizer.merges = {(a, b): a+b for (a, b) in data["merges"]}
        print(f"BPE 分词器已加载: vocab_size={tokenizer.vocab_size}, merges={len(tokenizer.merges)}")
        return tokenizer


# ─── 工厂函数 ─────────────────────────────────────────────────────────────────

def load_tokenizer(path: str):
    """自动检测并加载分词器类型"""
    config_path = os.path.join(path, "tokenizer.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到分词器文件: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    tok_type = data.get("type", "char")
    if tok_type == "bpe":
        return BPETokenizer.load(path)
    else:
        return CharTokenizer.load(path)


def build_tokenizer_from_files(
    data_files: List[str],
    save_path: str,
    tokenizer_type: str = "bpe",
    vocab_size: int = 8000,
    min_freq: int = 2,
) -> "BPETokenizer | CharTokenizer":
    """
    从数据文件构建并保存分词器

    Args:
        data_files: 文本文件列表
        save_path: 保存路径
        tokenizer_type: "bpe" 或 "char"
        vocab_size: 目标词表大小
        min_freq: 最小出现频率
    """
    texts = []
    for f in data_files:
        print(f"读取文件: {f}")
        with open(f, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                line = line.strip()
                if len(line) > 10:
                    texts.append(line)

    print(f"共读取 {len(texts):,} 行文本")

    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=vocab_size, min_freq=min_freq)
    else:
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(texts, min_freq=min_freq, max_vocab=vocab_size)

    tokenizer.save(save_path)
    return tokenizer
