import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def load_and_prepare_data(input_csv):
    df = pd.read_csv(input_csv)

    df['input_text'] = df['style'].apply(lambda s: f"<{s}> ") + df['abstract']
    df['target_text'] = df['title']

    train_df, val_df = train_test_split(df[['input_text', 'target_text']], test_size=0.1, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
    return dataset

if __name__ == "__main__":
    dataset = load_and_prepare_data("data/arxiv_abstracts_styles.csv")
    dataset.save_to_disk("data/arxiv_style_dataset")
