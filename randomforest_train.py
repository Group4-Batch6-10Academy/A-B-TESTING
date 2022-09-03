
from scripts.dataloader import DataLoader
from scripts.training_script import Training
from scripts.data_processor import DataProcessor
import numpy as np
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

import sys
import os


if __name__ == "__main__":
    # Data loading and Cleaning
    data_loader = DataLoader()
    df = data_loader.read_data('../data', 'AdSmartABdata.csv')

    data_processor = DataProcessor(df)
    df = data_processor.processes_data()
    df = data_processor.label_encoder()
    browser_data, platform_data = data_processor.split_data()

    # Trainging part
    data_trainer = Training(browser_data.drop(
        ["reaction", "auction_id"], axis=1), browser_data["reaction"])
    X_train, y_train, X_test, y_test, X_valid, y_valid = data_trainer.train_test_validate_split()
    model = RandomForestRegressor
    model = data_trainer.train(X_train, y_train)
    model = data_trainer.predict(model, X_test)
    data_trainer.feature_importance(model)
    data_trainer.confusion_matrix(model, X_test, y_test)
    data_trainer.classification_report(model, X_test, y_test)
    data_trainer.ROC_Curve(model, X_test, y_test)

    mlflow.log_metric(
        "basic score", data_trainer.basic_score(model, X_test, y_test))
    mlflow.log_metric("f1_score", data_trainer.f1_score(model, X_test, y_test))
    mlflow.sklearn.log_model(model, "logistic regression model")
    mlflow.log_artifact("Confusion_Matrix.png")
    mlflow.log_artifact("ROC_Curve.png")

    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
