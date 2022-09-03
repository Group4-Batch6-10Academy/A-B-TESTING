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


class Training():
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def train_test_validate_split(self):
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.X, self.y, train_size=0.8)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, train_size=0.875)

        return X_train, y_train, X_test, y_test, X_valid, y_valid

    def train(self, X_train, y_train):
        clf = self.model()
        clf.fit(X_train, y_train)
        self.model = clf
        return self.model

    def predict(self, X_test):
        self.model.predict(X_test)
        return self.model

    def feature_importance(self):
        if self.model.__class__.__name__ == "LogisticRegression":
            importance = self.model.coef_[0]
        else:
            importance = self.model.feature_importances_
        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i, v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.savefig("../images/Feature_Importance.png")
        plt.show()

    def basic_score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def recall(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return metrics.recall_score(y_test, y_pred)

    def f1_score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return metrics.f1_score(y_test, y_pred)

    def ROC_Curve(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
        auc = metrics.roc_auc_score(y_test, y_pred)
        plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.savefig("../images/ROC_Curve.png")
        plt.show()

    def confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_names = [0, 1]  # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix),
                    annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig("../images/Confusion_Matrix.png")
        plt.show()

    def classification_report(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        target_names = ['Yes', 'No']
        print(metrics.classification_report(
            y_test, y_pred, target_names=target_names))

    def hyperparameter_tunning(self, X_train, y_train):
        pipe = Pipeline([('classifier', RandomForestClassifier())])

        # GridSearchCV
        param_grid = [
            {'classifier': [self.model()],
             'classifier__penalty': ['l1', 'l2'],
             'classifier__C': np.logspace(-4, 4, 20),
             'classifier__solver': ['liblinear']},
            {'classifier': [RandomForestClassifier()],
             'classifier__n_estimators': list(range(10, 101, 10)),
             'classifier__max_features': list(range(6, 32, 5))}
        ]

        clf = GridSearchCV(pipe, param_grid=param_grid,
                           cv=5, verbose=True, n_jobs=-1)

        # Fitting the data

        best_fit = clf.fit(X_train, y_train)
        print(best_fit.param_grid)
        return best_fit
