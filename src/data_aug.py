import pandas as pd
import googletrans
import asyncio
from tqdm.asyncio import tqdm
from datasets import Dataset, DatasetDict, Value, concatenate_datasets
import random

CARENET_FILE = "../data/prepro/carenet_data_final.csv"
WELL_FILE = "../data/prepro/well_data_final.csv"
EMO_FILE = "../data/prepro/emotiontalk_final.csv"
EMER_FILE = "../data/prepro/emergency_final.csv"

TARGET_LABELS = ["positive", "danger", "critical", "emergency"]

TARGET_RATIO_MAP = {
    "positive": 0.9,  
    "danger": 0.8, 
    "critical": 1.0,   
    "emergency": 1.2
}

async def back_translate(text, translator):
    try:
        await asyncio.sleep(0.5)  # API 부하 방지
        translated_en_obj = await translator.translate(text, dest='en')
        back_translated_ko_obj = await translator.translate(translated_en_obj.text, dest='ko')
        return back_translated_ko_obj.text
    except Exception:
        return None

async def load_and_augment():
    translator = googletrans.Translator()
    all_final_dfs = []

    for input_file in [CARENET_FILE, WELL_FILE, EMO_FILE, EMER_FILE]:
        try:
            df = pd.read_csv(input_file)
            print(f"\n--- '{input_file}' 처리 시작 (원본 {len(df)}개) ---")
        except FileNotFoundError:
            print(f"!!! '{input_file}'을 찾을 수 없습니다.")
            continue

        label_counts = df["label"].value_counts().to_dict()
        max_count = max(label_counts.values())
        print(f"라벨 분포: {label_counts} (max={max_count})")

        augmented_rows = []

        for label, count in label_counts.items():
            if label not in TARGET_LABELS:
                continue

            target_ratio = TARGET_RATIO_MAP.get(label, 1.0)
            target_count = int(max_count * target_ratio)

            if count >= target_count:
                print(f"라벨 '{label}' 충분함 ({count}/{target_count}) → 증강 생략")
                continue

            augment_needed = target_count - count
            print(f"라벨 '{label}' 증강: {augment_needed}개 (목표 {target_count}, 현재 {count})")

            df_label = df[df["label"] == label].sample(n=min(count, augment_needed), replace=True)
            df_label.reset_index(drop=True, inplace=True)

            tasks = [back_translate(text, translator) for text in df_label["text"]]
            augmented_texts = await tqdm.gather(*tasks, desc=f"{label} 증강 중...")

            for orig, aug in zip(df_label["text"], augmented_texts):
                if aug and aug != orig:
                    augmented_rows.append({
                        "doll_id": df_label["doll_id"].iloc[0],
                        "text": aug,
                        "label": label
                    })

        df_aug = pd.DataFrame(augmented_rows)
        print(f"총 {len(df_aug)}개의 증강 샘플 생성 완료 ({input_file})")

        final_df = pd.concat([df, df_aug], ignore_index=True)
        all_final_dfs.append(final_df)

    combined_df = pd.concat(all_final_dfs, ignore_index=True)
    print(f"\n--- 최종 데이터 준비 완료 ---")
    print(f"전체 데이터 수: {len(combined_df)}")
    print("최종 라벨 분포:\n", combined_df["label"].value_counts())

    return combined_df

if __name__ == "__main__":
    import asyncio
    print("데이터 증강을 시작합니다. 데이터 양에 따라 시간이 다소 걸릴 수 있습니다...")
    final_augmented_df = asyncio.run(load_and_augment())
    raw_dataset = Dataset.from_pandas(final_augmented_df)
    raw_dataset = raw_dataset.cast_column("doll_id", Value("string"))

    dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    print("\n최종 데이터셋 정보:")
    print(dataset)

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    train_df.to_csv("../data/split/final_train_data.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv("../data/split/final_test_data.csv", index=False, encoding="utf-8-sig")

    final_augmented_df.to_csv("../data/final_augmented_data.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료: final_augmented_data.csv")
    print(f"전체: {len(final_augmented_df)}행")
    print(f" - 훈련 데이터: {len(train_df)}행 → final_train_data.csv")
    print(f" - 테스트 데이터: {len(test_df)}행 → final_test_data.csv")
