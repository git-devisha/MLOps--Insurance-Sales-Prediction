from sklearn.datasets import make_classification
import pandas as pd
import os

def extract_data():
    data_dir = os.path.join("insurance_sale", "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    os.makedirs(data_dir, exist_ok=True)

    append_mode = os.path.isfile(train_path)
    num_datasets = 10 if not append_mode else 1

    for i in range(num_datasets):
        x, y = make_classification(
            n_samples=10000,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42 + i  # Ensure different data in each iteration
        )

        df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        df['target'] = y

        train_data = df.iloc[:8000]
        test_data = df.iloc[8000:]

        train_data.to_csv(train_path, mode="a", header=not append_mode, index=False)
        test_data.to_csv(test_path, mode="w", header=True, index=False)  # Overwrite test.csv each time

    print("Extracted data from source successfully.")

if __name__ == "__main__":
    extract_data()
