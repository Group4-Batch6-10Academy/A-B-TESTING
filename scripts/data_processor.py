from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


class DataProcessor():
    def __init__(self, df):
        self.df = df

    def processes_data(self):
        control_df = self.df[self.df["experiment"] == "control"]
        exposed_df = self.df[self.df["experiment"] == "exposed"]

        yes_users_control = control_df[control_df["yes"] == 1]
        yes_users_exposed = exposed_df[exposed_df["yes"] == 1]
        no_users_control = control_df[control_df["no"] == 1]
        no_users_exposed = exposed_df[exposed_df["no"] == 1]

        # Drop yes, and no columns
        yes_users_control = yes_users_control.drop(['yes', 'no'], axis=1)
        yes_users_exposed = yes_users_exposed.drop(['yes', 'no'], axis=1)
        no_users_control = no_users_control.drop(['yes', 'no'], axis=1)
        no_users_exposed = no_users_exposed.drop(['yes', 'no'], axis=1)
        # Add a reaction column
        yes_users_control["reaction"] = 1
        yes_users_exposed["reaction"] = 1
        no_users_control["reaction"] = 0
        no_users_exposed["reaction"] = 0

        merged_control = pd.concat([yes_users_control, no_users_control])
        merged_exposed = pd.concat([yes_users_exposed, no_users_exposed])
        merged_df = pd.concat([merged_control, merged_exposed])

        # Shuffle rows
        merged_exposed = merged_exposed.sample(frac=1).reset_index(drop=True)
        merged_control = merged_control.sample(frac=1).reset_index(drop=True)
        merged_df = merged_df.sample(frac=1).reset_index(drop=True)

        self.df = merged_df
        return self.df

    def label_encoder(self):
        date_encoder = preprocessing.LabelEncoder()
        device_encoder = preprocessing.LabelEncoder()
        expt_encoder = preprocessing.LabelEncoder()
        browse_encoder = preprocessing.LabelEncoder()
        self.df["date"] = date_encoder.fit_transform(self.df["date"])
        self.df["experiment"] = expt_encoder.fit_transform(
            self.df["experiment"])
        self.df["device_make"] = device_encoder.fit_transform(
            self.df["device_make"])
        self.df["browser"] = browse_encoder.fit_transform(self.df["browser"])
        return self.df

    def split_data(self):
        browser_cols = ["auction_id", "experiment", "date",
                        "hour", "device_make", "browser", "reaction"]
        platform_cols = ["auction_id", "experiment", "date",
                         "hour", "device_make", "platform_os", "reaction"]
        df_browser = self.df[browser_cols]
        df_platform = self.df[platform_cols]
        return df_browser, df_platform
