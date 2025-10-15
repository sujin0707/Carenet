import pandas as pd

dataset3 = pd.read_json('./data/raw/talkemotion.json')

emotion_mapping = {
    # positive
    "E60": "positive", "E61": "positive", "E62": "positive", "E63": "positive",
    "E64": "positive", "E65": "positive", "E66": "positive", "E67": "positive",
    "E68": "positive", "E69": "positive",

    # danger
    "E20": "danger", "E21": "danger", "E23": "danger", "E27": "danger", "E28": "danger",
    "E30": "danger", "E31": "danger", "E32": "danger", "E33": "danger", "E34": "danger",
    "E35": "danger", "E36": "danger", "E37": "danger", "E38": "danger", "E39": "danger",
    "E10": "danger", "E11": "danger", "E12": "danger", "E13": "danger",
    "E14": "danger", "E15": "danger", "E16": "danger", "E17": "danger",
    "E18": "danger", "E19": "danger", "E22": "danger", "E24": "danger",
    "E25": "danger", "E26": "danger", "E29": "danger", "E40": "danger",
    "E41": "danger", "E42": "danger", "E43": "danger", "E45": "danger",
    "E46": "danger", "E47": "danger", "E50": "danger", "E51": "danger",
    "E52": "danger", "E53": "danger", "E54": "danger", "E55": "danger", "E56": "danger",
    "E57": "danger", "E58": "danger ", "E59": "danger",

    # critical
    "E44": "critical", "E48": "critical",
    "E49": "critical",
}


def preprocess_user_only_final(df, n=10):
    results = []

    for _, row in df.iterrows():
        emotion_id = row["profile"]["emotion"]["type"]
        risk_level = emotion_mapping.get(emotion_id, "unknown")

        talk_content = row["talk"]["content"]
        utterances = []
        for i in range(1, 10):
            hs = talk_content.get(f"HS{i:02d}", "")
            if hs:
                utterances.append(hs)

        if not utterances:
            continue

        if len(utterances) <= n:
            context = " ".join(utterances)
            results.append({
                "category": emotion_id,
                "text": context,
                "label": risk_level
            })
            continue
        
        stride = n // 2
        for i in range(n - 1, len(utterances), stride):
            window = utterances[i - n + 1 : i + 1]
            context = " ".join(window)

            results.append({
                "category": emotion_id,
                "text": context,
                "label": risk_level
            })

    df_out = pd.DataFrame(results)
    category_to_id = {cat: i + 521 for i, cat in enumerate(df_out["category"].unique())}
    df_out["doll_id"] = df_out["category"].map(category_to_id)

    df_out = df_out[["doll_id", "text", "label"]]
    return df_out

user_df = preprocess_user_only_final(dataset3, n=10)
print(user_df.head(10))

user_df.to_csv("./data/prepro/emotiontalk_final.csv", index=False, encoding="utf-8-sig")
print("Saved as emotiontalk_final.csv")
