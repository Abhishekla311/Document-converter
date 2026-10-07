import io
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import (LSTM,SimpleRNN, Dense,Input, Embedding, Dropout,  concatenate, TimeDistributed, RepeatVector, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from config.paths_config import *
from exception.CustomException import CustomException
from logger.logger import get_logger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

logger = get_logger(__name__)

class CNNFeatureExtractorModel:
    def __init__(self):
        logger.info("CNN Feature Extractoring")
        self.model = Sequential([
            Input(shape=(64,64,3)),
            Conv2D(32,(3,3), activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.2),
            Conv2D(16,(3,3), activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.2),
            Conv2D(8,(3,3), activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.2),
            Conv2D(4,(3,3,), activation="relu", kernel_regularizer=l2(0.001)),
            Flatten(),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization()
            
            ])
    def extract_features(self, image_path:str)->np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target Image missing {image_path}")
        img = image.load_image(image_path, target_path=(64,64, 3))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        features = self.model.predict(img_array, verbose=0)
        return features
class RNNOneToOne:
    def __init__(self, vocab_size:int, max_length:int):
        logger.info("RNNOneToMany FeatureExtracting")
        self.vocab_size = vocab_size
        self.max_length = max_length 
        EMBED_SIZE = 16
        HIDDEN_SIZE = 32


        self.model = Sequential([
            Input(shape=(1,)),
            Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE),
            SimpleRNN(units=HIDDEN_SIZE, return_sequences=True),
            Dense(units=self.vocab_size, activation="softmax"),
        ])
    
    def generate_text_one_to_one(self,feature_vector:np.ndarray, idx2word:dict)->str:
        prediction = self.model.predict(feature_vector, verbose=0)
        prediction_indices = np.argmax(prediction, axis=-1)[0]
        words = [idx2word[idx] for idx in prediction_indices if idx2word.get()!="<PAD>"]
        return " ".join(words)


class RnnOneToMany:
    def __init__(self, vocab_size:int, max_length:int):
        logger.info("RnnOneToMany Feature is extracting")
        self.vocab_size = vocab_size
        self.max_length = max_length
        EMBED_SIZE = 16
        HIDDEN_SIZE=32
        self.model = Sequential([
            Input(shape=(1,)),
            Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE),
            SimpleRNN(units=HIDDEN_SIZE, return_sequences=False),
            RepeatVector(max_length),
            SimpleRNN(units=vocab_size, return_sequences=True),
            TimeDistributed(Dense(units=self.vocab_size, activation="softmax"))
        ])

    def generate_one_to_many(self, feature_vector: np.ndarray, idx2word:dict)->str:
        prediction = self.model.predict(feature_vector, verbose=0),
        prediction_indices= np.argmax(prediction, axis=-1)[0]
        words = [idx2word[idx] for idx in  prediction_indices]
        return " ".join(words)

class RNNManyToOne:
    def __init__(self, vocab_size:int, max_length:int ):
        logger.info("RNNManyToOne feature is Extracting")
        self.vocab_size = vocab_size
        self.max_length = max_length
        EMBED_SIZE = 16
        HIDDEN_SIZE = 32
        self.model = Sequential([
            Input(shape=(self.max_length,)),
            Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE),
            SimpleRNN(units=HIDDEN_SIZE,return_sequences=False),
            Dense(units=self.vocab_size, activation="softmax")
        ])
    
    def generate_text_many_to_one(self, feature_vector:np.ndarray, idx2word:dict)->str:
        prediction = self.model.predict(feature_vector, verose=0)
        prediction_indices = np.argmax(prediction, axis=-1)[0]
        words = [idx2word[idx] for idx in prediction_indices]
        return " ".join(words).strip()
    

class RNNManyTOMany:
    def __init__(self, vocab_size:int, max_length:int, max_target_length:int):
        logger.info("RNNManyToOne feature is Extraction")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.max_target_length = max_target_length
        EMBED_SIZE = 16
        HIDDEN_SIZE = 32
        self.model = Sequential([
            Input(shape=(self.max_length,)),
            Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE),
            SimpleRNN(units=HIDDEN_SIZE, return_sequences=False),
            RepeatVector(self.max_target_length),
            SimpleRNN(units=HIDDEN_SIZE, return_sequences=True),
            TimeDistributed(Dense(units=self.vocab_size, activation="softmax"))
        ])

    def generate_text_many_to_many(self, feature_vector:np.ndarray, idx2word:dict)->str:
        logger.info("RNNManyToMany feature extacting")
        prediction = self.model.predict(feature_vector, verbose=0)
        prediction_indices = np.argmax(prediction, axis=-1)[0]
        words = [idx2word[idx] for idx in prediction_indices]
        return " ".join(words).strip()

class CombineCNNRNN:
    def __init__(self, vocab_size:int, max_length:int):
        logger.info("CombineCNNRNN feature is extracting")
        self.max_length = max_length
        self.vocab_size = vocab_size
        EMBED_SIZE = 16
        HIDDEN_SIZE = 32
        text_input = Input(shape=(self.max_length, ), name="text_input")
        x_text = Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE)(text_input)
        x_text = LSTM(32, return_sequences=True)(x_text)
        x_text = Dropout(0.001)(x_text)
        x_text = LSTM(16, return_sequences=False)(x_text)        #  Fixed: x_test -> x_text
        text_features = Dense(8, activation="relu")(x_text)  

        image_input = Input(shape=(64, 64, 3), name="image_input")
        x_img = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(0.001))(image_input)
        x_img = BatchNormalization()(x_img)
        x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
        x_img = Dropout(0.2)(x_img)
        
        x_img = Conv2D(16, (3, 3), activation="relu", kernel_regularizer=l2(0.001))(x_img)
        x_img = BatchNormalization()(x_img)
        x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
        x_img = Dropout(0.2)(x_img)
        
        x_img = Conv2D(8, (3, 3), activation="relu", kernel_regularizer=l2(0.001))(x_img)
        x_img = BatchNormalization()(x_img)
        x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
        x_img = Dropout(0.2)(x_img)
        
        x_img = Conv2D(4, (3, 3), activation="relu", kernel_regularizer=l2(0.001))(x_img)
        x_img = Flatten()(x_img)
        x_img = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x_img)
        x_img = BatchNormalization()(x_img) 
        x_img = Dropout(0.002)(x_img)
        image_features = Dense(16, activation="relu", kernel_regularizer=l2(0.001))(x_img)

        combined_features = concatenate([text_features, image_features])
        combined_dense = Dense(16, activation="relu")(combined_features)
        final_output =  Dense(1, activation="sigmoid" )(combined_dense)
        self.model = Model(inputs=[text_input, image_input ], outputs=final_output)
    
    def predict_combine(self, text_tensor:np.ndarray, image_tensor:np.ndarray)->float:
        prediction = self.model.predict({"text_input":text_tensor, "image_input":image_tensor })
        return float(prediction[0][0])

class ModelRegistry:
    _instances = {}
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(ModelRegistry, cls).__new__(cls)
        return cls.instance

    def get_cnn_model(self):
        if "cnn" not in self._instances:
            self._instances["cnn"] = CNNFeatureExtractorModel()
        return self._instances["cnn"]

    # 1. यहाँ self के आगे vocab_size, max_length जोड़ें:
    def get_one_to_one_model(self, vocab_size, max_length):
        if "one_to_one" not in self._instances:
            self._instances["one_to_one"] = RNNOneToOne(vocab_size, max_length)
        return self._instances["one_to_one"]
    
    # 2. यहाँ self के आगे vocab_size, max_length जोड़ें:
    def get_one_to_many(self, vocab_size, max_length):
        if "one_to_many" not in self._instances:
            self._instances["one_to_many"] = RnnOneToMany(vocab_size, max_length)
        return self._instances["one_to_many"]

    # 3. यहाँ self के आगे vocab_size, max_length जोड़ें:
    def get_many_to_one(self, vocab_size, max_length):
        if "many_to_one" not in self._instances:
            self._instances["many_to_one"] = RNNManyToOne(vocab_size, max_length)
        return self._instances["many_to_one"]
    
    # 4. यहाँ self के आगे vocab_size, max_length, max_target_length जोड़ें:
    def get_many_to_many(self, vocab_size, max_length, max_target_length):
        if "many_to_many" not in self._instances:
            self._instances["many_to_many"] = RNNManyTOMany(vocab_size, max_length, max_target_length)
        return self._instances["many_to_many"]

    # 5. यहाँ self के आगे vocab_size, max_length जोड़ें:
    def get_combinecnnrnn(self, vocab_size, max_length):
        if "combine_model" not in self._instances:
            self._instances["combine_model"] = CombineCNNRNN(vocab_size, max_length)
        return self._instances["combine_model"]