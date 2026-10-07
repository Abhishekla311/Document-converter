import os 
import pandas as pd
import io
import urllib.request
import pymongo
import requests
from PIL import Image

from dotenv import load_dotenv
load_dotenv()
from logger.logger import get_logger
from config.paths_config  import *
from exception.CustomException import CustomException
logger = get_logger(__name__)
class StudentDataIngestion:

  
  def __init__(self, config:dict):
    
    self.config = config["student_data_ingestion"]
    self.mongo_url = os.getenv("uri")
    if not self.mongo_url:
      raise CustomException("Mongodb URL is not found in the environment variables")
    

    self.db_name = self.config["db_name"]

    self.collection_name = self.config["collection_name"]

    self.config_bucket_name = self.config["bucket_name"]




    self.region = self.config["region"]
   
    os.makedirs(RAW_DIR, exist_ok = True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    logger.info(f"Data ingestion started with {self.config_bucket_name} and file is {self.collection_name}")

  def download_csv_from_mongodb(self)->pd.DataFrame:
    try:
      from pymongo import MongoClient
      client = MongoClient(self.mongo_url)
      db = client[self.db_name]
      collection = db[self.collection_name]
      df = collection.find()
      df = pd.DataFrame(df)

      if df is None  or len (df)==0:
        raise CustomException(f"No records found in the collection {self.collection_name}", Exception("Database collection is empty"))
      else:
        if "_id" in df.columns:
          df.drop("_id", axis=1, inplace=True)
        df.to_csv(RAW_FILE_PATH, index=False)
      
      return df
    except Exception as e:
      logger.error("Error while downloading the csv from mongodb")
      raise CustomException("Failed to download csv from mongodb", e)
  def download_images_from_s3(self, df:pd.DataFrame): 
    try:
        logger.info("Starting the image downloading process")
        for _, row in df.iterrows():
           name = row.get("name", "Unknown")
           photo_url = row.get("photo_url", None)
           if not photo_url:
             continue

           response = requests.get(photo_url, stream=True, timeout=15)
           if response.status_code ==200:
                image = Image.open(io.BytesIO(response.content))
              
                clean_images = f"{name.lower().replace(' ', '_')}.jpg"
                image.save(os.path.join(IMAGES_DIR, clean_images))
           else:
                logger.warning(f"Failed to download image for {name} from {photo_url}.status_code :{response.status_code}")
    except Exception as e:
        logger.error("Error while downloading the csv from mongodb")
        if isinstance(e, CustomException):
           raise e
        raise CustomException("Failed to download images from s3", e)

  def run(self):
    try:
        df = self.download_csv_from_mongodb()
        self.download_images_from_s3(df)
        logger.info("Data ingestion process completed successfully")

    except Exception as e:
        logger.error("Error occurred while running the data ingestion process")
        if isinstance(e, CustomException):
                   raise e
        raise CustomException("Failed to run data ingestion", e)


if __name__ == "__main__":
    from utils.common_functions import read_yaml
    config_data = read_yaml(CONFIG_PATH)
    ingestion = StudentDataIngestion(config_data)
    ingestion.run()



