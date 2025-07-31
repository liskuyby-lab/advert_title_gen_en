import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline
from models.generate import generate_title

def load_test_data(test_csv):
    df = pd.read_csv(test_csv)
    return df

def evaluate_generation(test_df):
    generated = []
    for _, row in test_df.iterrows():
        gen_title = generate_title(row['abstract'], row['style'])
        generated.append(gen_title)

    test_df['generated_title'] = generated

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = ["short", "humorous", "standard"]

    predicted_styles = []
    for title in generated:
        result = classifier(title, candidate_labels)
        predicted_styles.append(result['labels'][0])

    acc = accuracy_score(test_df['style'], predicted_styles)
    print(f"Style classification accuracy: {acc:.3f}")

    return acc

if __name__ == "__main__":
    test_df = load_test_data("data/arxiv_test.csv")
    evaluate_generation(test_df)
