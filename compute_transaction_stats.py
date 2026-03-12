"""
Compute rolling and anomaly statistics for enriched transactions.

Features added:
- Rolling mean/std for amount by sender and recipient
- IQR-based outlier flags
- Mahalanobis distance of amount along with balance_after
- Entropy of transaction descriptions per user
- KL divergence between sender/recipient amount distributions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from scipy.stats import iqr, entropy, wasserstein_distance


def load_enriched():
    file = Path('transactions_enriched_with_locations.csv')
    if not file.exists():
        raise FileNotFoundError(str(file))
    return pd.read_csv(file)


def add_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # rolling by sender_name
    df = df.sort_values('timestamp')
    df['amount_numeric'] = pd.to_numeric(df['amount'], errors='coerce')
    df['balance_numeric'] = pd.to_numeric(df['balance_after'], errors='coerce')

    for prefix in ['sender', 'recipient']:
        grp = df.groupby(f'{prefix}_name')['amount_numeric']
        df[f'{prefix}_roll_mean'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{prefix}_roll_std'] = grp.transform(lambda x: x.rolling(window, min_periods=1).std())
        df[f'{prefix}_roll_median'] = grp.transform(lambda x: x.rolling(window, min_periods=1).median())
    return df


def add_iqr_outlier(df: pd.DataFrame) -> pd.DataFrame:
    amounts = df['amount_numeric'].dropna()
    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)
    iqr_val = q3 - q1
    lower = q1 - 1.5 * iqr_val
    upper = q3 + 1.5 * iqr_val
    df['amount_iqr_outlier'] = ~df['amount_numeric'].between(lower, upper)
    return df


def mahalanobis_distance(df: pd.DataFrame) -> pd.DataFrame:
    # compute using amount and balance
    vals = df[['amount_numeric', 'balance_numeric']].dropna()
    cov = np.cov(vals.values, rowvar=False)
    invcov = np.linalg.pinv(cov)
    mean = vals.mean().values
    distances = []
    for idx, row in df[['amount_numeric', 'balance_numeric']].iterrows():
        if pd.isna(row).any():
            distances.append(np.nan)
            continue
        diff = row.values - mean
        distances.append(mahalanobis(diff, np.zeros(len(diff)), invcov))
    df['mahalanobis'] = distances
    return df


def description_entropy(df: pd.DataFrame) -> pd.DataFrame:
    # compute entropy of description text for each sender
    def ent(series):
        text = " ".join(series.dropna().astype(str))
        probs = pd.Series(list(text)).value_counts(normalize=True)
        return entropy(probs, base=2)
    ent_map = df.groupby('sender_name')['description'].transform(ent)
    df['sender_desc_entropy'] = ent_map
    return df


def kl_divergence(df: pd.DataFrame) -> pd.DataFrame:
    # approximate by comparing amount distribution of sender vs recipient
    divs = []
    for idx, row in df.iterrows():
        s = df[df['sender_name'] == row['sender_name']]['amount_numeric'].dropna()
        r = df[df['recipient_name'] == row['recipient_name']]['amount_numeric'].dropna()
        # create histograms
        if len(s) < 2 or len(r) < 2:
            divs.append(np.nan)
            continue
        bins = np.histogram_bin_edges(pd.concat([s, r]), bins='auto')
        p, _ = np.histogram(s, bins=bins, density=True)
        q, _ = np.histogram(r, bins=bins, density=True)
        # add small epsilon to avoid zeros
        p += 1e-9
        q += 1e-9
        divs.append(entropy(p, qk=q))
    df['kl_divergence'] = divs
    return df


if __name__ == '__main__':
    df = load_enriched()
    df = add_rolling_stats(df)
    df = add_iqr_outlier(df)
    df = mahalanobis_distance(df)
    df = description_entropy(df)
    df = kl_divergence(df)

    # save enhanced stats
    df.to_csv('transactions_with_stats.csv', index=False)
    print('Saved transactions_with_stats.csv with statistics.')
    
    # print some stats summary
    print('Summary of new columns:')
    print(df[['sender_roll_mean','recipient_roll_mean','amount_iqr_outlier','mahalanobis','sender_desc_entropy','kl_divergence']].describe())
