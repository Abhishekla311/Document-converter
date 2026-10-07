import os
from venv import logger
import boto3
import pymongo
from dotenv import load_dotenv
from logger.logger import get_logger
from exception.CustomException import CustomException


load_dotenv()

class CloudImagePipeline:
    def __inti__(self, config):
            self.mongo_url = os.getenv("mongo_uri")
            self.client = pymongo.MongoClient(self.mongo_url)
            self.db = self.client["student_data"]
            self.collection = self.db["student_images"]
            self.bucket_name = config["bucket_name"]
            self.region = config["region"]
            self.s3_client = boto3.client(
            "s3", aws_access_key_id = os.get("aws_access_key_id"), aws_secret_access_key = os.getenv("aws_secret_access_key"), region_name = self.region)
    
    def upload_image_to_s3(self, image_path, student_details):
        try:
            file_name = os.path.basename(image_path)
            print(f"uploading  {file_name} to AWS to s3 ...")


            self.s3_client.upload_file(image_path, self.bucket_name, file_name)
            s3_url = f"https://{self.bucket_name}.s3.{self.region}://{file_name}"
            logger.info(f"Image uploaded to S3 successfully. URL:{s3_url}")
            student_details["image_url"] = s3_url
            result = self.collection.insert_one(student_details)
            logger.info(f"Student details inserted into MongoDB successfully. ID: {result.inserted_id}")
        except Exception as e:
            raise CustomException("Failed to upload image to s3", e)