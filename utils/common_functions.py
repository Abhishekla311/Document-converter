import os
import yaml
import pandas as pd
from  logger.logger import get_logger
from exception.CustomException import CustomException
logger = get_logger(__name__)
def read_yaml(file_path:str)->dict:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        with open(file_path,"r") as file:
            config = yaml.safe_load(file)
            logger.info("Successfully read the Yaml file ")
            return config
    except Exception as e:
        logger.error("Error while reading Yaml file")
        raise CustomException("Failed to read Yaml file", e)
    
    