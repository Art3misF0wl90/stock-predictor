# app/ml/splitter.py
#
# Time-safe train / validation / test splitting.
#
# WHY time-based splitting matters:
# Standard random splits would let the model train on 2022 data and then
# "predict" on 2019 data — effectively seeing the future.  These functions
# split strictly by row position so the test set is always the most recent
# data and there is zero temporal leakage between splits.
#
# Split ratios are read from config.py (default 70/15/15).

import numpy as np

from config import TRAIN_RATIO, VAL_RATIO


def time_split(
    df,
    feature_cols: list[str],
) -> tuple[tuple, tuple, tuple]:
    """
    Split a feature DataFrame into (train, val, test) tuples chronologically.

    Returns three (X, y) tuples:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)

    The DataFrame must already have a "target" column.
    Rows are never shuffled; the first TRAIN_RATIO fraction goes to train,
    the next VAL_RATIO fraction to val, and the remainder to test.
    """
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X = df[feature_cols].values
    y = df["target"].values

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(
        f"  Split → Train: {len(X_train)} "
        f"| Val: {len(X_val)} "
        f"| Test: {len(X_test)}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D feature array into 3-D sequences for LSTM input.

    For each position i >= seq_len, the input sequence is X[i-seq_len : i]
    and the target is y[i].  This means the first seq_len rows are consumed
    as context and do not produce a prediction, so the returned arrays are
    shorter than the input by seq_len rows.

    Returns
    -------
    X_seq : shape (n_samples, seq_len, n_features)
    y_seq : shape (n_samples,)
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)
