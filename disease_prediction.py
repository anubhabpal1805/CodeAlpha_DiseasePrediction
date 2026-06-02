import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==============================
# Configuration
# ==============================

DATASET_PATH = "heart.csv"
MODEL_PATH = "disease_model.pkl"
RANDOM_STATE = 42


# ==============================
# Functions
# ==============================

def load_data():
    """
    Load the heart disease dataset.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        print("Dataset loaded successfully.")
        return df

    except FileNotFoundError:
        print(f"Error: '{DATASET_PATH}' not found.")
        return None


def prepare_data(df):
    """
    Separate features and target.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def train_model(X_train, y_train):
    """
    Train Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL RESULTS ==========")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save results to file
    with open("results.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.2%}\n\n")
        file.write("Classification Report\n")
        file.write(classification_report(y_test, predictions))

    return predictions


def save_model(model):
    """
    Save trained model.
    """
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("\nModel saved successfully.")


def create_confusion_matrix(y_test, predictions):
    """
    Generate confusion matrix visualization.
    """
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved.")


def create_feature_importance(model, X):
    """
    Generate feature importance visualization.
    """
    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))

    plt.barh(
        X.columns,
        importance
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig("feature_importance.png")
    plt.close()

    print("Feature importance graph saved.")


def main():

    df = load_data()

    if df is None:
        return

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(model)

    create_confusion_matrix(
        y_test,
        predictions
    )

    create_feature_importance(
        model,
        X
    )

    print("\nProject executed successfully.")


if __name__ == "__main__":
    main()
