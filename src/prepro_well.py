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

dataset2 = pd.read_excel('./data/raw/well.xlsx')

prefix_mapping = {
    "감정/걱정": "danger",
    "감정/감정조절이상" : "critical",
    "감정/고독감" : "critical",
    "감정/공포" : "critical",
    "감정/공허감" : "danger",
    "감정/과민반응" : "critical",
    "감정/괴로움" : "critical",
    "감정/기분저하" : "danger",
    "감정/기시감" : "danger",
    "감정/긴장" : "danger",
    "감정/눈물" : "critical",
    "감정/답답" : "danger",
    "감정/당황" : "danger",
    "감정/두려움" : "critical",
    "감정/멍함" : "danger",
    "감정/모호함" : "danger",
    "감정/무력감" : "emergency",
    "감정/무미건조" : "danger",
    "감정/무서움" : "critical",
    "감정/미안함/자녀" : "danger",
    "감정/미움" : "critical",
    "감정/배신감" : "danger",
    "감정/부정적사고" : "emergency",
    "감정/분노" : "critical",
    "감정/불만" : "danger",
    "감정/불신" : "danger",
    "감정/불안감" : "danger",
    "감정/불쾌감" : "critical",
    "감정/불편감" : "danger",
    "감정/비관적" : "critical",
    "감정/살인욕구" : "emergency",
    "감정/생각" : "danger",
    "감정/서운함" : "danger",
    "감정/속상함" : "danger",
    "감정/슬픔" : "danger",
    "감정/신경쓰임" : "danger",
    "감정/심란" : "critical",
    "감정/억울함" : "danger",
    "감정/예민함" : "danger",
    "감정/외로움" : "danger",
    "감정/우울감" : "critical",
    "감정/의기소침" : "danger",
    "감정/의욕상실" : "danger",
    "감정/자괴감" : "danger",
    "감정/자살충동" : "emergency",
    "감정/자신감저하" : "danger",
    "감정/자존감저하" : "danger",
    "감정/절망감" : "critical",
    "감정/좌절" : "danger",
    "감정/죄책감" : "danger",
    "감정/짜증" : "critical",
    "감정/창피함" : "danger",
    "감정/초조함" : "danger",
    "감정/충격" : "danger",
    "감정/통제력상실" : "critical",
    "감정/허무함" : "critical",
    "감정/화" : "critical",
    "감정/후회" : "danger",
    "감정/힘듦" : "critical",
    "배경/가족" : "danger",
    "배경/건강문제" : "danger",
    "배경/결혼" : "danger",
    "배경/경제적문제" : "danger",
    "배경/공부/부진" : "danger",
    "배경/남편" : "danger",
    "배경/대인관계" : "danger",
    "배경/문제" : "danger",
    "배경/부모" : "danger",
    "배경/부모/아버지/폭력" : "critical",
    "배경/사고" : "danger",
    "배경/생활/폭행/피해" : "danger",
    "배경/시댁" : "danger",
    "상태/증상지속" : "danger",
    "원인/없음" : "danger",
    "자가치료/심리조절" : "danger",
    "증상/기억력저하" : "critical",
    "증상/두근거림" : "critical",
    "증상/두통" : "critical",
    "증상/무기력" : "critical",
    "증상/반복사고" : "critical",
    "증상/반복행동" : "critical",
    "증상/불면" : "danger",
    "증상/식욕저하" : "critical",
    "증상/죽음공포" : "emergency",
    "증상/통증" : "critical",
    "증상/폭식" : "danger",
    "증상/피로" : "danger",
    "증상/피해망상" : "emergency",
    "증상/호흡곤란" : "emergency",

}

def map_label(category: str) -> str:
    for prefix, label in prefix_mapping.items():
        if category.startswith(prefix):
            return label
    return "positive"

dataset2["위험도"] = dataset2["구분"].apply(map_label)

danger_labels = [
    "감정/곤혹감", "내원이유/상담", "배경/남자친구/고민/없음", "배경/음주/알코올의존", "배경/이혼", "배경/자각/우울증", "배경/자각/정신질환",
    "배경/자녀", "배경/직장/고민/퇴사", "배경/직장/복직", "배경/직장/불만", "배경/직장/스트레스", "배경/직장/퇴사", "배경/직장/휴직",
    "배경/타인/갈등", "배경/학교/갈등/선생님", "배경/학교/따돌림", "증상/건강염려", "증상/만성피로", "증상/생리불순",
    "증상/소화불량", "증상/속쓰림", "증상/저림현상/발/손", "증상/집중력저하", "증상/체력저하", "증상/편두통", "현재상태/증상지속",
    "배경/남자친구/없음", "배경/남자친구/이별", "배경/남자친구/집착", "배경/남자친구/짧은교제", "배경/부모", "배경/부모/죽음", "배경/사업/경제적문제/실패",
    "배경/사업/실패", "배경/생활/불가능/운전", "배경/생활/스트레스", "배경/생활/혼자", "배경/성격/예민함", "배경/애완동물/가족/갈등", "배경/어린시절/가난",
    "배경/여자친구/관계소원", "배경/여자친구/이별", "배경/육아/힘듦", "배경/임신/낙태", "배경/직장/불만/업무", "배경/취업/힘듦", "모호함"
]
critical_labels = [
    "내원이유/의사소견", "내원이유/치료", "증상/가슴떨림", "증상/가슴통증", "증상/공격적성향", "증상/공황발작", "증상/과대망상", "증상/과수면",
    "증상/대인기피", "증상/두근거림", "증상/떨림", "증상/메스꺼움", "증상/시력저하", "증상/신체이상/목", "증상/악몽", "증상/가슴답답",
    "증상/어지러움", "증상/은둔", "증상/이명", "증상/이인감", "증상/인지기능저하", "증상/체중감소", "증상/체중증가", "증상/힘빠짐", "치료이력/병원내원",
    "치료이력/병원내원/복약", "치료이력/응급실", "현재상태/증상악화"

]

emergency_labels = ["증상/기억상실", "증상/기절", "증상/기절예기", "증상/발작", "증상/알코올의존", "증상/자살시도",
                    "증상/자해", "증상/환각", "증상/환청",
                    ]

label_mapping = {
    **{lbl: "danger" for lbl in danger_labels},
    **{lbl: "critical" for lbl in critical_labels},
    **{lbl: "emergency" for lbl in emergency_labels}
}

# 후처리
def refine_label(row):
    if row["위험도"] == "positive":
        return label_mapping.get(row["구분"], "positive")
    return row["위험도"]

dataset2["위험도"] = dataset2.apply(refine_label, axis=1)

positive_labels = [
    "배경/남편/관계양호", "배경/남편/사업", "배경/남편/의지",
    "배경/대인관계/양호",
    "배경/부모/어머니/죽음", "배경/부모/이혼", "배경/생활/양호", "배경/생활/여행/해외", "배경/생활/운동",
    "배경/생활/자연소멸/증상"
]

def map_positive(category: str, current_label: str) -> str:
    if category in positive_labels:
        return "positive"
    return current_label 

dataset2["위험도"] = dataset2.apply(
    lambda row: map_positive(row["구분"], row["위험도"]), axis=1
)

dataset2 = dataset2[~dataset2["구분"].isin(["부가설명", "일반대화"])]

print(dataset2)

def get_recent_n(df, n=10, stride=None):
    results = []

    priority = {
        "positive": 1,
        "danger": 2,
        "critical": 3,
        "emergency": 4
    }

    for category, group in df.groupby("구분"):
        group = group.reset_index(drop=True)
        length = len(group)

        if stride is None:
            current_stride = n // 2 if length > n else max(1, length // 2)
        else:
            current_stride = min(stride, max(1, length // 2))

        if length <= n:
            utterances = group["유저"].tolist()
            context = " ".join(utterances)

            max_label = group["위험도"].map(priority).max()
            label = [k for k, v in priority.items() if v == max_label][0]

            results.append({
                "category": category,
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
            utterances = window["유저"].tolist()
            context = " ".join(utterances)

            max_label = window["위험도"].map(priority).max()
            label = [k for k, v in priority.items() if v == max_label][0]

            results.append({
                "category": category,
                "text": context,
                "label": label
            })

    return pd.DataFrame(results)

windowed_df = get_recent_n(dataset2, n=10)
category_to_id = {cat: i+164 for i, cat in enumerate(windowed_df["category"].unique())}
windowed_df["doll_id"] = windowed_df["category"].map(category_to_id)

df = windowed_df[["doll_id", "text", "label"]]
print(df.head())

df.to_csv("./data/prepro/well_data_final.csv", index=False, encoding="utf-8-sig")
