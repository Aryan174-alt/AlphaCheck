"""Tests for AI Lab modules: no sentence overlap, tensor shapes, and tagger determinism."""

import numpy as np
import pytest
import torch
from alphacheck.deep_models import SmallTCNClassifier, SmallTransformerClassifier
from alphacheck.llm_tagger import LocalLLMEventTagger
from alphacheck.sentiment_finetune import load_and_split_phrasebank


def test_no_sentence_overlap_in_splits():
    train_df, val_df, test_df = load_and_split_phrasebank(seed=42, sample_size=100)

    train_sentences = set(train_df["sentence"])
    val_sentences = set(val_df["sentence"])
    test_sentences = set(test_df["sentence"])

    # Zero overlap guaranteed
    assert len(train_sentences.intersection(val_sentences)) == 0
    assert len(train_sentences.intersection(test_sentences)) == 0
    assert len(val_sentences.intersection(test_sentences)) == 0


def test_deep_sequence_model_tensor_shapes():
    batch_size = 4
    seq_len = 20
    input_dim = 8

    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # 1. Transformer
    trans = SmallTransformerClassifier(input_dim=input_dim, d_model=16, nhead=2, num_layers=1)
    out_trans = trans(dummy_input)
    assert out_trans.shape == (batch_size,)

    # 2. TCN
    tcn = SmallTCNClassifier(input_dim=input_dim, num_channels=[16, 16], kernel_size=3)
    out_tcn = tcn(dummy_input)
    assert out_tcn.shape == (batch_size,)


def test_llm_tagger_deterministic(tmp_path):
    cache_file = tmp_path / "test_llm_cache.json"
    tagger = LocalLLMEventTagger(cache_path=str(cache_file))

    h1 = "Reliance reports 25% surge in Q3 net profit"
    h2 = "SEBI issues regulatory notice over disclosure norms"

    t1_first = tagger.tag_headline(h1)
    t1_second = tagger.tag_headline(h1)

    t2_first = tagger.tag_headline(h2)

    assert t1_first == "earnings"
    assert t1_second == "earnings"
    assert t2_first == "regulatory"

