"""A2 AI Lab: Dense Headline Embeddings & PCA Reduction.

Extracts dense 384-dimensional sentence embeddings using sentence-transformers
('all-MiniLM-L6-v2') and reduces them via PCA to a compact set of components (e.g. 5).
Aligned strictly before 09:00 AM IST.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class HeadlineEmbeddingExtractor:
    """Extracts dense sentence representations from news headlines."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_components: int = 5):
        self.model_name = model_name
        self.n_components = n_components
        self.model = None
        self.pca = None

    def load_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def extract_and_reduce(
        self,
        headlines: List[str],
        fit_pca: bool = True,
    ) -> np.ndarray:
        """Encode headlines into embeddings and reduce to n_components via PCA."""
        if not headlines:
            return np.zeros((0, self.n_components))

        self.load_model()
        embeddings = self.model.encode(headlines, batch_size=32, show_progress_bar=False)

        if fit_pca or self.pca is None:
            self.pca = PCA(n_components=min(self.n_components, len(headlines)), random_state=42)
            reduced = self.pca.fit_transform(embeddings)
        else:
            reduced = self.pca.transform(embeddings)

        # Pad to self.n_components if smaller
        if reduced.shape[1] < self.n_components:
            pad = np.zeros((reduced.shape[0], self.n_components - reduced.shape[1]))
            reduced = np.hstack([reduced, pad])

        return reduced

