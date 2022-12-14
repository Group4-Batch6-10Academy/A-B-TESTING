
from scripts.dataloader import DataLoader
from scripts.training_script import Training
from scripts.data_processor import DataProcessor
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import mlflow
import mlflow.sklearn

import sys
import os


def xgboost_train(data):
    """A function to train xgboost model
        Args:
            data- Prepared data for training
    """
    model = GradientBoostingRegressor
    data_trainer = Training(data.drop(
        ["reaction", "auction_id"], axis=1), data["reaction"], model=model)
    X_train, y_train, X_test, y_test, X_valid, y_valid = data_trainer.train_test_validate_split()

    model = data_trainer.train(X_train, y_train)
    pred = data_trainer.predict(X_test)
    data_trainer.feature_importance()
    # data_trainer.confusion_matrix(X_test, y_test)
    # data_trainer.classification_report(X_test, y_test)
    # data_trainer.ROC_Curve(X_test, y_test)

    mlflow.log_metric(
        "basic score", data_trainer.basic_score(X_test, y_test))
    #mlflow.log_metric("f1_score", data_trainer.f1_score(X_test, y_test))
    mlflow.sklearn.log_model(model, "logistic regression model")
    mlflow.log_artifact("../images/Confusion_Matrix.png")
    mlflow.log_artifact("../images/ROC_Curve.png")

    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
