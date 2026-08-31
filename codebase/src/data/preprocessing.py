"""Preprocessing for test failure prediction text."""

from __future__ import annotations
from typing import List
import pandas as pd


class TestPreprocessor:
    def __init__(self, max_text_length: int = 800):
        self.max_text_length = max_text_length

    def clean(self, text: str) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = " ".join(str(text).split())
        return text[: self.max_text_length]

    def transform(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def transform_df(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        df = df.copy()
        df["cleaned_text"] = self.transform(df[text_col].tolist())
        return df
