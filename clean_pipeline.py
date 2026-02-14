# Importing my pipeline minus the comments into a new file, to ensure a smooth import
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def cc_pipeline(df):
    """
        Pipeline for College Completion dataset.
        This function will serve as a structure for both pipelines
    """

    # Copy of course, can't lose it
    df = df.copy()

    # 1.) Fix variable types- converting columns when necessary (categorical, numerical)
    categorical_cols = [
        "level",
        "control",
        "hbcu",
        "flagship",
    ]
    for col in categorical_cols:  # converting to categorical type
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2.) Dropping non-predictive columns and strictly identifiers
    # Reason: These columns wont help the model predict specific outcomes I'm looking for
    id_cols = [
        "index",
        "unitid",
        "chronname",
        "city",
        "nicknames",
        "site",
        "similar",
        "vsa_grad",
        "vsa_enroll"
    ]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])
    df = df.drop(columns=["state", "basic", "counted_pct"], errors="ignore")

    # 3.) Identifying target col and dropping rows with missing target values
    # Target column will be grad 150 value
    target_col = "grad_150_value"

    # Drop rows where target_col is missing
    df = df.dropna(subset=[target_col])

    # Convert continuous graduation rate into binary classification
    # 1 = Above median graduation rate
    # 0 = Below or equal to median
    median_val = df[target_col].median()
    df["grad_flag"] = (df[target_col] > median_val).astype(int)

    # Drop original continuous target
    df = df.drop(columns=[target_col])

    # Assign predictors and target
    Y = df["grad_flag"]
    X = df.drop(columns=["grad_flag"])
    # Now that x and y separated ready for one-hot encoding and scaling

    # 4.) One-hot encoding categorical variables to create binary columns
    # Identify cat columns in variable x
    cat_cols = list(X.select_dtypes(include=["category", "object"]))
    # Apply one-hot encoding to categorical columns
    X = pd.get_dummies(X, columns=cat_cols)

    # 6.) Train, Tune, Test split
    # First split vs remaining data
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X,
        Y,
        test_size=0.4,
        random_state=42,
        stratify=Y
    )

    # Second split into tune and test
    X_tune, X_test, Y_tune, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=0.5,
        random_state=42,
        stratify=Y_temp
    )

    # 7.) Now impute missing values after splitting
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_tune = pd.DataFrame(
        imputer.transform(X_tune),
        columns=X_tune.columns,
        index=X_tune.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # 8.) Scale AFTER imputation
    scaler = MinMaxScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_tune = pd.DataFrame(
        scaler.transform(X_tune),
        columns=X_tune.columns,
        index=X_tune.index
    )

    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Returning all datasets
    return X_train, X_tune, X_test, Y_train, Y_tune, Y_test


#%%
# Testing the pipeline on the original dataset to ensure it runs without errors and produces expected outputs
if __name__ == "__main__":
    df = pd.read_csv("Data/cc_institution_details.csv")

    X_train, X_tune, X_test, Y_train, Y_tune, Y_test = cc_pipeline(df)

    print("Train shape:", X_train.shape)
    print("Tune shape:", X_tune.shape)
    print("Test shape:", X_test.shape)

    print("Train class mean:", Y_train.mean())
    print("Tune class mean:", Y_tune.mean())
    print("Test class mean:", Y_test.mean())

    print("Missing values in train:", X_train.isna().sum().sum())

# %%
