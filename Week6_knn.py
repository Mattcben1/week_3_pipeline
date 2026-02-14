
# %%
# Importing necessary libraries, including my pipeline function from question 4
import pandas as pd
from clean_pipeline import cc_pipeline 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
#%%

# 1:
# - I had to convert my target variable to a classification target.
# - I also went ahead and made a clean file containing my pipeline function, which I imported here.
# - In doing so, I minimize errors and ensure that my data is clean and ready for modeling.

#%%
# 2: Build a KNN=3 model

# Loading the data 
df = pd.read_csv("Data/cc_institution_details.csv")

# Also running our pipeline function to ensure the data is clean and ready for knn
X_train, X_tune, X_test, Y_train, Y_tune, Y_test = cc_pipeline(df)

# Building the KNN model with equal to 3
knn = KNeighborsClassifier(n_neighbors=3)

# Fitting the model on the training data
knn.fit(X_train, Y_train)

# Predicting it on test set
Y_pred = knn.predict(X_test)

# producing probabilities for positive class
Y_prob = knn.predict_proba(X_test)[:, 1]

# %%
X_train.isna().sum().sum()

# %%
# 3: Creating the Dataframe
results_df = pd.DataFrame({
    'Actual': Y_test,
    'Predicted': Y_pred,
    'Probability': Y_prob
})
print(results_df.head())

# %%

#Question 4:  

# If you adjusted the k hyperparameter what do you think would
# happen to the threshold function? Would the confusion matrix look the same at the same threshold 
# levels or not? Why or why not?

# Answer: Since K represents the number of nesrest neighbors, increasing it 
#  Could lead to smoother probability estimates, while decreasing it could lead to more volatile estimates.
#  In return, the threshold function would be affected, outputing differen classifications at the same threshold levels
#  which would lead to a different confusion matrix.

# 5:
cm = confusion_matrix(Y_test, Y_pred)
print("CM:", cm)

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
print("A:", accuracy)

# In regards to my question, "Can we predict whether a college has an above median graduation rate, 150%
# When 1 is assigned to above median and 0 is assigned to below median.
#         0    1
#    0  [324   24]
#    1  [27   319]
#  324 colleges were correctly classified as having a below median graduation rate (TN)
#  24 colleges were incorrectly classified as having an above median graduation rate (FP)
#  27 colleges were incorrectly classified as having a below median graduation rate (FN)
#  319 colleges correctlty classified as having an above median graduation rate (TP)
#  Taking into account the accuracy of this output, the confusion matrix seems be be balanced 
# Positives: Accuracy, Low FP and FN, High TP
# Concerns: sensitive to noise, unseen structures could cause misclassifications

# %%
# 6: 
# Funtion one has been implemented earlier in the notebook, but for organization purposes I will display it.
# %%
def run_knn_model(X_train, Y_train, X_test, Y_test, k=3, threshold=0.5):
    """
    KNN Classification Training and Evaluation Function

        k = # of nearest neighbors
        threshold = cutoff of predicting positive class

    Returns:
        confusion matrix and accuracy 
    """

    # Building the KNN model using chosen k value
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fitting model based on training data
    knn.fit(X_train, Y_train)

    # Generating predicted probabilities for positive class
    Y_prob = knn.predict_proba(X_test)[:, 1]

    # custom threshold to determine final predictions
    Y_pred = (Y_prob >= threshold).astype(int)

    # confusion matrix to evaluate performance
    cm = confusion_matrix(Y_test, Y_pred)

    #  overall accuracy
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()

    return cm, accuracy


# %%
# %%
# This function consists of testing multiple k and threshold combinations to optimize our models' performance

k_values = [1, 3, 5, 7, 9]
threshold_values = [0.3, 0.4, 0.5, 0.6]

for k in k_values:
    for t in threshold_values:

        # KNN model with chosen k and threshold
        cm, acc = run_knn_model(
            X_train, Y_train,
            X_test, Y_test,
            k=k,
            threshold=t
        )
        # Printing performance metrics 
        print("k =", k,
              "| threshold =", t,
              "| accuracy =", round(acc, 4))

# %%

# 7: How well does the model perform? Did the interaction of the adjusted thresholds and k values help the model? Why or why not? 

# Answer: K and the threshold values interaction did help the model by allowing us to find the optimal balance between sensitivity and specificity.
# For example, a lower threshold (0.3) with a smaller k (1) would lead to higher sensitivity but lower specificity, 
# while a higher threshold (0.6) with a larger k (9) would lead to higher specificity but lower sensitivity.
# Overall, the best performance was k=9 with a threshold of 0.6, yeilding a 93% accuracy, which is a one percent improvement from our initial model with k=3 and threshold of 0.5.

#%%

# 8:

# %%
from sklearn.model_selection import train_test_split

def cc_pipeline_control(df):
    """
        Using the same function as before, but switching the target variable to 'control'.
        This version predicts whether an institution is Public or Private.
    """

    # Copy of course, can't lose it
    df = df.copy()

    # 1.) Fix variable types- converting columns when necessary (categorical, numerical)
    categorical_cols = [
        "level",
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

    # 3.) Identifying target col
    # Target column will be control (Public vs Private)

    # Converting control column into binary classification
    # 1 = Public institution
    # 0 = Private institution
    df["control"] = df["control"].astype(str).str.strip().str.upper()
    df["control"] = df["control"].apply(
        lambda x: 1 if "PUBLIC" in x else 0
    )

    # Assign predictors and target
    Y = df["control"]
    X = df.drop(columns=["control"])
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

    # 7.) Impute missing values AFTER splitting
    from sklearn.impute import SimpleImputer
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
    from sklearn.preprocessing import MinMaxScaler
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

# Using the same testing function as before to evaluate the model with the new target variable
X_train, X_tune, X_test, Y_train, Y_tune, Y_test = cc_pipeline_control(df)

cm, acc = run_knn_model(
    X_train, Y_train,
    X_test, Y_test,
    k=3,
    threshold=0.5
)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", acc)

# Intepretation of 8: 
# The model perfromed extremely well, with an accuracy of 98.5%. 
# The confusion matrix shows that the model correctly classified 309 public institutions (TP) 
# and 441 private institutions (TN), with only 3 misclassifications (FP and FN).

# %%
