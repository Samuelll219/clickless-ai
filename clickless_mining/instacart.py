"""Instacart 2017 data -> category-level transactions, plus train/test splits.

Files (Kaggle "instacart-market-basket-analysis", kept OUT of git because of
the dataset terms and size): orders.csv, products.csv, aisles.csv,
order_products__prior.csv, order_products__train.csv. The `test` eval set has
no products and is ignored.

A transaction is the set of catalog leaf categories in one order, after
mapping product -> aisle -> leaf (`aisles.mapping`) and dropping unmapped
lines. Categories are kept in the order they were first added to the cart
(`add_to_cart_order`), which the "first item" Hit@k protocol uses.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from clickless_kg import vocab

LEAVES = tuple(vocab.CATEGORY_NAMES)
LEAF_INDEX = {leaf: i for i, leaf in enumerate(LEAVES)}
ORDER_PRODUCT_FILES = ("order_products__prior.csv", "order_products__train.csv")


@dataclass
class Transactions:
    """One row per order with >= 1 mapped category."""
    frame: pd.DataFrame          # order_id, user_id, order_number, eval_set, categories (tuple[str])
    stats: dict

    def baskets(self) -> list[tuple[str, ...]]:
        return list(self.frame["categories"])


def build_transactions(instacart_dir: str | Path, aisle_to_leaf: dict[int, str],
                       files=ORDER_PRODUCT_FILES) -> Transactions:
    d = Path(instacart_dir)
    products = pd.read_csv(d / "products.csv", usecols=["product_id", "aisle_id"],
                           dtype={"product_id": np.int32, "aisle_id": np.int16})
    lookup = np.full(int(products["product_id"].max()) + 1, -1, dtype=np.int8)
    aisle_leaf = {a: LEAF_INDEX[leaf] for a, leaf in aisle_to_leaf.items()}
    leaf_of_aisle = products["aisle_id"].map(aisle_leaf).fillna(-1).astype(np.int8).to_numpy()
    lookup[products["product_id"].to_numpy()] = leaf_of_aisle

    parts, n_lines = [], 0
    for name in files:
        op = pd.read_csv(d / name, usecols=["order_id", "product_id", "add_to_cart_order"],
                         dtype={"order_id": np.int32, "product_id": np.int32,
                                "add_to_cart_order": np.int16})
        n_lines += len(op)
        op["leaf"] = lookup[op["product_id"].to_numpy()]
        parts.append(op.loc[op["leaf"] >= 0, ["order_id", "leaf", "add_to_cart_order"]])
    lines = pd.concat(parts, ignore_index=True)
    n_mapped_lines = len(lines)
    first = (lines.groupby(["order_id", "leaf"], sort=False)["add_to_cart_order"].min()
             .reset_index().sort_values(["order_id", "add_to_cart_order", "leaf"]))
    cats = first.groupby("order_id", sort=True)["leaf"].agg(lambda s: tuple(LEAVES[i] for i in s))

    orders = pd.read_csv(d / "orders.csv",
                         usecols=["order_id", "user_id", "eval_set", "order_number"],
                         dtype={"order_id": np.int32, "user_id": np.int32,
                                "order_number": np.int16, "eval_set": "category"})
    n_orders_with_products = int(orders["eval_set"].isin(["prior", "train"]).sum())
    frame = orders.merge(cats.rename("categories"), left_on="order_id", right_index=True)
    frame = frame.sort_values(["user_id", "order_number"]).reset_index(drop=True)
    sizes = frame["categories"].map(len)
    stats = {
        "order_lines": int(n_lines), "mapped_order_lines": int(n_mapped_lines),
        "mapped_line_share": n_mapped_lines / n_lines if n_lines else 0.0,
        "orders_with_products": n_orders_with_products,
        "orders_with_ge1_category": int(len(frame)),
        "orders_with_ge2_categories": int((sizes >= 2).sum()),
        "mean_categories_per_order": float(sizes.mean()) if len(frame) else 0.0,
        "users": int(frame["user_id"].nunique()),
    }
    return Transactions(frame, stats)


def save(tx: Transactions, path: str | Path) -> None:
    pd.to_pickle({"frame": tx.frame, "stats": tx.stats}, path)


def load(path: str | Path) -> Transactions:
    obj = pd.read_pickle(path)
    return Transactions(obj["frame"], obj["stats"])


def split_by_user(frame: pd.DataFrame, test_frac: float = 0.2, seed: int = 573):
    """All orders of a random `test_frac` of users are held out (no user leakage)."""
    users = sorted(frame["user_id"].unique().tolist())
    rng = random.Random(seed)
    test_users = set(rng.sample(users, round(len(users) * test_frac)))
    mask = frame["user_id"].isin(test_users)
    return frame[~mask].reset_index(drop=True), frame[mask].reset_index(drop=True)


def split_by_order(frame: pd.DataFrame, test_frac: float = 0.2, seed: int = 573):
    """Random 20% of orders (the proposal's literal wording); users overlap."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(frame)) < test_frac
    return frame[~mask].reset_index(drop=True), frame[mask].reset_index(drop=True)
