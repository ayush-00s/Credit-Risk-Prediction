import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

## where data will be saved :
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    ## reading data
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ## reading data from source
            df = pd.read_csv('notebook/data/cs-training.csv')
            logging.info("Read the dataset as dataframe")

            ## drop the unwanted index column from the raw CSV
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)

            ## remove rows where age == 0 (data entry errors)
            df.drop(df[df["age"] == 0].index, inplace=True)

            ## remove rows with special/placeholder values (96, 98) in delinquency columns
            delinquency_cols = [
                "NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
                "NumberOfTime60-89DaysPastDueNotWorse"
            ]
            for col in delinquency_cols:
                df = df[~df[col].isin([96, 98])]

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["SeriousDlqin2yrs"]
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))