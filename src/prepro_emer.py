import pandas as pd

def preprocess_emergency(df, start_id=528):
    df["category"] = "emergency_group"

    category_to_id = {cat: i + start_id for i, cat in enumerate(df["category"].unique())}
    df["doll_id"] = df["category"].map(category_to_id)

    df_out = df[["doll_id", "text", "label"]]
    return df_out

emergency_df = pd.read_csv("./data/raw/emergency.csv")
final_emergency = preprocess_emergency(emergency_df, start_id=165)

print(final_emergency)
final_emergency.to_csv("./data/prepro/emergency_final.csv", index=False, encoding="utf-8-sig")
print("Saved as emergency_final.csv")
