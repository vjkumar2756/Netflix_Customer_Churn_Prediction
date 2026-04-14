import sys
import os
import pickle

# FIX PATH (strong fix)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

from data_preprocessing import load_data, preprocess_data
from model_training import train_logistic, train_knn
from evaluation import evaluate_model

def main():
    df = load_data(os.path.join(BASE_DIR, "data/net.csv"))

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    lr_model = train_logistic(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    lr_acc, _ = evaluate_model(lr_model, X_test, y_test)
    knn_acc, _ = evaluate_model(knn_model, X_test, y_test)

    print("Logistic Regression Accuracy:", lr_acc)
    print("KNN Accuracy:", knn_acc)

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    best_model = lr_model if lr_acc > knn_acc else knn_model

    with open(os.path.join(BASE_DIR, "models/model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(BASE_DIR, "models/scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("Best model saved successfully!")

if __name__ == "__main__":
    main()