"""CSV schema helpers for uploaded review datasets."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

TEXT_ALIASES = (
    "text",
    "review",
    "review_text",
    "content",
    "body",
    "tweet_text",
    "tweet text",
    "feedback",
    "comment",
)

SCORE_ALIASES = (
    "score",
    "rating",
    "stars",
    "star_rating",
    "overall",
    "review_score",
)

PRODUCT_ALIASES = (
    "product",
    "product_name",
    "product name",
    "author",
    "username",
    "user",
)

TIME_ALIASES = (
    "time",
    "date",
    "created_at",
    "created at",
    "tweet_created_at",
    "tweet created at",
    "timestamp",
)


def normalize_column_name(column: object) -> str:
    """Return a comparable form for a CSV header."""
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def find_column(columns: Iterable[object], aliases: Iterable[str]) -> object | None:
    """Find the first column whose normalized name matches an alias."""
    normalized_aliases = {normalize_column_name(alias) for alias in aliases}

    for column in columns:
        if normalize_column_name(column) in normalized_aliases:
            return column

    return None


def prepare_uploaded_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a dataframe with the app's required Text and Score columns."""
    text_column = find_column(df.columns, TEXT_ALIASES)
    if text_column is None:
        raise ValueError("CSV must include a review text column.")

    result = df.copy()
    result["Text"] = result[text_column].fillna("").astype(str).str.strip()
    result = result[result["Text"] != ""].copy()

    if result.empty:
        raise ValueError("CSV text column has no review rows.")

    score_column = find_column(result.columns, SCORE_ALIASES)
    used_default_score = score_column is None
    if score_column is None:
        result["Score"] = 3
    else:
        result["Score"] = pd.to_numeric(result[score_column], errors="coerce").fillna(3)

    product_column = find_column(result.columns, PRODUCT_ALIASES)
    if product_column is not None and "Product" not in result.columns:
        result["Product"] = result[product_column]

    time_column = find_column(result.columns, TIME_ALIASES)
    if time_column is not None and "Time" not in result.columns:
        result["Time"] = result[time_column]

    return result.reset_index(drop=True), {
        "text_column": text_column,
        "score_column": score_column,
        "used_default_score": used_default_score,
    }
