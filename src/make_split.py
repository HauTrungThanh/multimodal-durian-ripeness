from sklearn.model_selection import train_test_split

import pandas as pd

def safe_group_split(fruit_id: pd.DataFrame, train_ratio: float, val_ratio:float, test_ratio: float, seed: int):
    labels = fruit_id["label"].values
    stratify = labels if pd.Series(labels).value_counts().min() >= 2 else None
    
    train_fruits, temp_fruits = train_test_split(
        fruit_id,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    
    ramaining = val_ratio + test_ratio
    val_size_in_temp = val_ratio / ramaining if ramaining > 0 else 0.5
    temp_labels = temp_fruits["label"].values
    temp_stratify = temp_labels if pd.Series(temp_labels).value_counts().min() >= 2 else None

    val_fruits, test_fruits = train_test_split(
        temp_fruits,
        train_size=val_size_in_temp,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )
    return train_fruits, val_fruits, test_fruits

def main():
    ...

    