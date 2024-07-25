import pandas as pd
import datasets
from sklearn.model_selection import train_test_split

def get_raw_data(csv_name, label_col, text_col):
    df = pd.read_csv(f"data/{csv_name}")
    if text_col != "text":
        df["text"] = df[text_col]
    df["label"] = df[label_col]
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"""\nData breakdown by label:{df[label_col].value_counts()}""")
    return df[['text','label']]

def train_val_test_split(df, label_col_name="label", train_size=0.8, has_val=True):
    """Return a tuple (DataFrame, DatasetDict) with a custom train/val/split"""

    if isinstance(train_size, int):
        train_size = train_size / len(df)

    df = df.sample(frac=1, random_state=0)
    df_train, df_test = train_test_split(
        df, train_size=train_size, stratify=df[label_col_name], random_state=333
    )

    if has_val:
        df_test, df_val = train_test_split(
            df_test, test_size=0.5, stratify=df_test[label_col_name]
        )
        return (
            (df_train, df_val, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "val": datasets.Dataset.from_pandas(df_val),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )

    else:
        return (
            (df_train, df_test),
            datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_pandas(df_train),
                    "test": datasets.Dataset.from_pandas(df_test),
                }
            ),
        )
    