import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# import the following classes upon cpmpletion of data transformation
from src.components.data_transformation import DataTransformation # check functionality at the bottom
from src.components.data_transformation import DataTransformationConfig # check functionality at the bottom

# import the following classes upon completion of model training
from src.components.model_trainer import ModelTrainerConfig 
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig: # this class creates paths for data ingestion
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")     # defining file paths
    raw_data_path: str=os.path.join('artifact',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #calling@DataIngestionClass activates DataIngestionConfig into ingestion_config class

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the raw data as a dataframe') #keeping track of loggings to spot exceptions more easily

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #saves csv file to raw_data_path folder

            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.inititate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr)) #printing this gives r2_score
    
