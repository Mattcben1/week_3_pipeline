# %%
# College Data Pipeline 


# Importing all libraries needed for this project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from io import StringIO 

# %%
# Importing dataset and calling shape
df = pd.read_csv("Data/cc_institution_details.csv")
df.shape
#%%
pd.set_option("display.max_rows", None)
df.dtypes
# %%
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
    for col in categorical_cols: #converting to categorical type 
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2.)Dropping non-predictive columns and stricly identifiers
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

    # 4.) One-hot encoding categorical variables to create binary colomuns
    # Identify cat colomuns in variable x
    cat_cols = list(X.select_dtypes(include=["category", "object"]))
    # Apply one-hot encoding to categorical columns
    X = pd.get_dummies(X, columns=cat_cols)

    # 5.) Scaling numerical features on a comparable scale
    # 0-1 range to ensure no numerical domination outside these parameters 
    num_cols = list(X.select_dtypes(include="number"))
    X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    # 6.) Train, Tune, Test split
    # Frist Split vs remaining data
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
    # Returning all datasets
    return X_train, X_tune, X_test, Y_train, Y_tune, Y_test
    

# %%
# Assigning outputs of pipeline to call the pipeline itself
X_train, X_tune, X_test, Y_train, Y_tune, Y_test = cc_pipeline(df)

# %%
print(Y_train.value_counts())
#%%
X_train.shape
#%%
# Checking and comparing the x and the y rows of our sets
# If my pipeline is correct, the rows should match
print(X_train.shape, Y_train.shape)
print(X_tune.shape, Y_tune.shape)
print(X_test.shape, Y_test.shape)




#%%
# Job Placement Dataset 

# %%
jp_df = pd.read_csv("Data/job_placement.csv")
jp_df.shape

# %%
def jp_pipeline(jp_df): # Utilizing funtions from my first piepline and adjusting for the job placement dataset
    """
    Job Placement Dataset Pipeline.
    Returns train, tune, and test splits.
    """
    # Copy of course, can't lose it
    jp_df = jp_df.copy()

    # 1.) Fix variable types- converting columns when necessary (categorical, numerical)
    categorical_cols = [
        "gender",
        "ssc_b",
        "hsc_b",
        "hsc_s",
        "degree_t",
        "workex",
        "specialisation",
    ]
    for col in categorical_cols: #converting to categorical type 
        if col in jp_df.columns:
            jp_df[col] = jp_df[col].astype("category")

    # 2.)Dropping non-predictive columns and stricly identifiers
    # Reason: These colums are missing lots of data
    jp_df = jp_df.drop(columns=["sl_no", "salary"], errors="ignore")

    # 3.) Identifying target col and dropping rows with missing target values
    jp_df["status"] = jp_df["status"].map({"Placed": 1, "Not Placed": 0})

    # Drop rows where target_col is missing
    jp_df = jp_df.dropna(subset=["status"])

    # assigning predictors x and y
    Y = jp_df["status"]
    X = jp_df.drop(columns=["status"])
    # Now that x and y separated ready for one-hot encoding and scaling

    # 4.) One-hot encoding categorical variables to create binary colomuns
    # Identify cat colomuns in variable x
    cat_cols = list(X.select_dtypes(include=["category", "object"]))
    # Apply one-hot encoding to categorical columns
    X = pd.get_dummies(X, columns=cat_cols)

    # 5.) Scaling numerical features on a comparable scale
    # 0-1 range to ensure no numerical domination outside these parameters 
    num_cols = list(X.select_dtypes(include="number"))
    X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    # 6.) Train, Tune, Test split
    # Frist Split vs remaining data
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
        random_state=42
        stratify=Y_temp
    )
    # Returning all datasets
    return X_train, X_tune, X_test, Y_train, Y_tune, Y_test
# %%
# Assigning outputs of pipeline to call the pipeline itself
# Using the same testing block as last time
# Again, if my pipeline is correct, the rows should match
X_train, X_tune, X_test, y_train, y_tune, y_test = jp_pipeline(jp_df)

print(X_train.shape, y_train.shape)
print(X_tune.shape, y_tune.shape)
print(X_test.shape, y_test.shape)

