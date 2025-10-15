import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
import optuna
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import platform
import transformers
import joblib
import pandas as pd

dataset = pd.read_excel('./data/raw/carenet.xlsx')
dataset["label"] = dataset["label"].astype(str).str.strip().str.lower()

def get_recent_n(df, n=10, stride=None):
    """
    n: window size
    stride: window step
    """
    results = []

    priority = {
        "positive": 1,
        "danger": 2,
        "critical": 3,
        "emergency": 4
    }

    unmatched_labels = set()

    for doll_id, group in df.groupby("doll_id"):
        group = group.sort_values("uttered_at").reset_index(drop=True)
        length = len(group)

        if stride is None:
            current_stride = n // 2 if length > n else max(1, length // 2)
        else:
            current_stride = min(stride, max(1, length // 2))

        if length <= n:
            utterances = group["text"].tolist()
            context = " ".join(utterances)

            mapped = group["label"].map(priority)
            if mapped.isnull().any():
                unmatched = group.loc[mapped.isnull(), "label"].unique()
                unmatched_labels.update(unmatched)
                continue

            max_label = mapped.max()
            label = [k for k, v in priority.items() if v == max_label][0]

            results.append({
                "doll_id": doll_id,
                "text": context,
                "label": label
            })
            continue

        windows = []
        for start in range(0, length - n + 1, current_stride):
            end = start + n
            windows.append((start, end))

        if windows[-1][1] < length:
            windows.append((length - n, length))

        for start, end in windows:
            window = group.iloc[start:end]
            utterances = window["text"].tolist()
            context = " ".join(utterances)

            mapped = window["label"].map(priority)
            if mapped.isnull().any():
                unmatched = window.loc[mapped.isnull(), "label"].unique()
                unmatched_labels.update(unmatched)
                continue

            max_label = mapped.max()
            label = [k for k, v in priority.items() if v == max_label][0]

            results.append({
                "doll_id": doll_id,
                "text": context,
                "label": label
            })

    if unmatched_labels:
        print("매칭되지 않은 라벨 :", unmatched_labels)

    return pd.DataFrame(results)

windowed_df = get_recent_n(dataset, n=10)
print(windowed_df.head())

windowed_df.to_csv("./data/prepro/carenet_data_final.csv", index=False, encoding="utf-8-sig")
