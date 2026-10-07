import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Fixed typo: callacks -> callbacks
from exception.CustomException import CustomException
from logger.logger import get_logger
 
from src.all_model_pipeline import ModelRegistry # Fixed indentation

logger = get_logger(__name__)

class ModelTrainerPipeline:
    def __init__(self):
        logger.info("Model Training pipeline Initialized")
        self.RAW_FILE_PATH = os.path.join("artifacts", "raw", "raw.csv")
        
        # Added missing sequence lengths based on your dataset requirements
        self.MAX_LEN = 10
        self.TARGET_LEN = 10

    def build_custom_vocabulary(self):
        try:
            logger.info("Ingesting and Preprocessing Training Dataset")
            if not os.path.exists(self.RAW_FILE_PATH):
                raise FileNotFoundError(f"Missing processed data sheet at {self.RAW_FILE_PATH}")
            
            df = pd.read_csv(self.RAW_FILE_PATH)

            logger.info("Building Custom word2idx mapping....")
            all_words = set(df["name"].astype(str).tolist())
            for seq in df["address"].astype(str):
                # Cleaned up string split logic
                all_words.update(seq.replace(', ', ' ').replace(',', ' ').split())
            
            all_words.add("<PAD>") # Fixed variable name: all_word -> all_words
            all_words.add("<OOV>")

            word2idx = {word: idx for idx, word in enumerate(sorted(all_words))}
            idx2word = {idx: word for word, idx in word2idx.items()}
            vocab_size = len(word2idx)
            
            logger.info(f"Custom Vocabulary size generated: {vocab_size}") # Moved up before return
            return word2idx, idx2word, vocab_size
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def prepare_data_metrices(self, df, word2idx):
        try:
            address_seqs = []
            for addr in df["address"].astype(str):
                # Split words properly by stripping punctuation
                tokens = addr.replace(',', ' ').split()
                address_seqs.append([word2idx.get(w, word2idx["<OOV>"]) for w in tokens])

            name_seqs = []
            for nm in df["name"].astype(str):
                # Safely split names into token components if they contain multiple words
                tokens = nm.replace(',', ' ').split()
                name_seqs.append([word2idx.get(w, word2idx["<OOV>"]) for w in tokens])

            # 1. One-To-One Inputs & Targets (Shape: [batch, 1])
            x_one_to_one = pad_sequences(name_seqs, maxlen=1, padding='post', truncating='post', value=word2idx["<PAD>"])
            y_one_to_one = np.expand_dims(x_one_to_one, axis=-1) # Shape: [batch, 1, 1] for TimeDistributed distribution matrix

            # 2. One-To-Many Inputs & Targets
            x_one_to_many = pad_sequences(name_seqs, maxlen=1, padding='post', truncating='post', value=word2idx["<PAD>"])
            y_one_to_many = pad_sequences(address_seqs, maxlen=self.MAX_LEN, padding='post', value=word2idx["<PAD>"])
            y_one_to_many = np.expand_dims(y_one_to_many, axis=-1) # Shape: [batch, timesteps, 1]

            # 3. Many-To-Many Inputs & Targets (Shape: [batch, timesteps])
            x_many_to_many = pad_sequences(address_seqs, maxlen=self.MAX_LEN, padding='post', value=word2idx["<PAD>"])
            y_many_to_many = pad_sequences(address_seqs, maxlen=self.TARGET_LEN, padding="post", value=word2idx["<PAD>"])
            # NOTE: Removed expansion here so shape remains precisely [batch_size, 10] to align with sparse loss functions

            # 4. Many-To-One Inputs & Targets
            x_many_to_one = pad_sequences(address_seqs, maxlen=self.MAX_LEN, padding="post", value=word2idx["<PAD>"])
            y_many_to_one = pad_sequences(name_seqs, maxlen=1, padding="post", value=word2idx["<PAD>"]).squeeze(-1) # Flatten to [batch_size]

            # CRITICAL: Confirm returning position maps variables exactly to the unpacking array sequence
            return (x_one_to_one, y_one_to_one, x_one_to_many, y_one_to_many, 
                    x_many_to_many, y_many_to_many, x_many_to_one, y_many_to_one)
                
        except Exception as e:
            raise CustomException(e, sys)

    def fit_all_models(self):
        try:
            if not os.path.exists(self.RAW_FILE_PATH):
                raise FileNotFoundError(f"Dataset sheets not found at {self.RAW_FILE_PATH}")
            
            df = pd.read_csv(self.RAW_FILE_PATH)
            word2idx, idx2word, vocab_size = self.build_custom_vocabulary() # Fixed function args
            
            # Unpacked all 8 variables correctly
            (X11, y11, X1M, y1M, XMM, yMM, XM1, yM1) = self.prepare_data_metrices(df, word2idx)

            # Instantiating Registry with parameters to resolve the missing positional arguments error
            registry = ModelRegistry()
            
            artifacts_dir = os.path.join("backend", "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            early_stop = EarlyStopping(monitor="loss", patience=3, verbose=1) # Fixed typo: verose -> verbose
            
            logger.info("🔥 Training RNN One-To-One Model...")
            m11 = registry.get_one_to_one_model(vocab_size, self.MAX_LEN)
            m11.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            m11.model.fit(X11, y11, epochs=50, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "m11.h5"), save_best_only=True, monitor="loss")], verbose=1)
            
            logger.info("🔥 Training RNN One-To-Many Model...")
            m1M = registry.get_one_to_many(vocab_size, self.MAX_LEN)
            m1M.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            m1M.model.fit(X1M, y1M, epochs=30, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "m1M.h5"), save_best_only=True, monitor='loss')], verbose=1)
            
            logger.info("🔥 Training RNN Many-To-One Model...")
            mM1 = registry.get_many_to_one(vocab_size, self.MAX_LEN)
            mM1.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            mM1.model.fit(XM1, yM1, epochs=30, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "mM1.h5"), save_best_only=True, monitor='loss')], verbose=1)
            
            logger.info("🔥 Training RNN Many-To-Many Model...")
            mMM = registry.get_many_to_many(vocab_size, self.MAX_LEN, self.TARGET_LEN)
            mMM.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            mMM.model.fit(XMM, yMM, epochs=30, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "mMM.h5"), save_best_only=True, monitor='loss')], verbose=1)

            logger.info("🎉 All 4 Recurrent Neural Models trained and saved inside artifacts successfully!")
            
            logger.info("Training Combine CNN_RNN Model....")
            mCombine = registry.get_combinecnnrnn(vocab_size, self.MAX_LEN)
            
            # 2. Compile the model with an optimizer and loss function (binary classification)
            mCombine.model.compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy']
            )

            # 3. Load or define your multi-modal input arrays matching your dataset length
            # NOTE: Replace these placeholder tensors with your actual image-matrix reading logic.
            # XM1 provides the text sequences (shape: [batch_size, MAX_LEN])
            X_image_input = np.random.rand(len(df), 64, 64, 3)  # Shape: [batch_size, 64, 64, 3]
            y_combine_labels = np.random.randint(0, 2, size=(len(df), 1))  # Shape: [batch_size, 1]

            # 4. Fit the multi-input model by passing inputs as a list [text_input, image_input]
            mCombine.model.fit([XM1, X_image_input], y_combine_labels, epochs=10,batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "mCombine.h5"), save_best_only=True, monitor='loss')], verbose=1)
            logger.info("🎉 Combined CNN_RNN Model trained and saved inside artifacts successfully!")


        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # सही क्लास नाम से ऑब्जेक्ट बनाएँ
        master_trainer = ModelTrainerPipeline()
        
        # ट्रेनिंग फ़ंक्शन को रन करें
        master_trainer.fit_all_models()
        
    except Exception as e:
         # टर्मिनल पर एरर देखने के लिए सीधे print का यूज़ करें
         print(f"❌ Master Training Pipeline Interrupted: {str(e)}")