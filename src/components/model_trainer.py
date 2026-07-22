import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # imbalance ratio, used for XGBoost's scale_pos_weight
            neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
            scale_pos_weight = neg / pos
            logging.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

            models = {
                "Random Forest Classifier (balanced)": RandomForestClassifier(
                    class_weight='balanced', random_state=42
                ),
                "XG Boost Classifier (balanced)": XGBClassifier(
                    scale_pos_weight=scale_pos_weight, random_state=42
                ),
            }

            params = {
                "Random Forest Classifier (balanced)": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "max_features": ["sqrt", "log2"],
                },
                "XG Boost Classifier (balanced)": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                },
            }

            model_report, fitted_models = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # best model based on ROC-AUC (appropriate for this imbalanced dataset)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = fitted_models[best_model_name]

            logging.info(f"Best model: {best_model_name} | ROC-AUC: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable ROC-AUC score", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_test_pred = best_model.predict(X_test)
            y_test_proba = best_model.predict_proba(X_test)[:, 1]

            final_roc_auc = roc_auc_score(y_test, y_test_proba)
            final_f1 = f1_score(y_test, y_test_pred)
            final_precision = precision_score(y_test, y_test_pred)
            final_recall = recall_score(y_test, y_test_pred)
            final_accuracy = accuracy_score(y_test, y_test_pred)

            logging.info(
                f"Final test metrics — Accuracy: {final_accuracy:.4f}, "
                f"F1: {final_f1:.4f}, Precision: {final_precision:.4f}, "
                f"Recall: {final_recall:.4f}, ROC-AUC: {final_roc_auc:.4f}"
            )

            return final_roc_auc

        except Exception as e:
            raise CustomException(e, sys)