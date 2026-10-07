import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# आपके प्रोजेक्ट के कस्टम एक्सेप्शन और लॉगर मॉड्यूल्स
from exception.CustomException import CustomException
from logger.logger import get_logger
from all_model_pipeline import ModelRegistry 

logger = get_logger(__name__)

class CustomModelTrainer:
    def __init__(self):
        logger.info("🎬 Custom Multi-Modal Training Pipeline Initialized...")
        self.RAW_FILE_PATH = os.path.join("backend", "artifacts", "raw", "raw.csv")
        self.MAX_LENGTH = 10 # पते के लिए अधिकतम शब्दों की लंबाई

    def load_and_prepare_data(self):
        try:
            logger.info("📥 Ingesting and Preprocessing Dataset...")
            
            # 1. CSV डेटा लोड करना
            if not os.path.exists(self.RAW_FILE_PATH):
                raise FileNotFoundError(f"Missing processed data sheet at {self.RAW_FILE_PATH}")
            df = pd.read_csv(self.RAW_FILE_PATH)

            # =========================================================
            # YOUR CUSTOM VOCABULARY PROCESSING LOGIC
            # =========================================================
            logger.info("🔤 Building Custom Word2Idx Mapping...")
            all_words = set(df["name"].astype(str).tolist()) 
            for seq in df["address"].astype(str):
                all_words.update(seq.replace(',', '').split()) 
            all_words.add("<PAD>") 
            all_words.add("<OOV>") # Out of Vocabulary टोकन बैकअप के लिए

            word2idx = {word: idx for idx, word in enumerate(sorted(all_words))}
            idx2word = {idx: word for word, idx in word2idx.items()}
            vocab_size = len(word2idx)
            logger.info(f"📚 Custom Vocabulary Size Generated: {vocab_size}")

            # 2. टेक्स्ट पते को न्यूमेरिकल सीक्वेंसेस में बदलना (Text to Matrix)
            text_sequences = []
            for addr in df["address"].astype(str):
                cleaned_addr = addr.replace(',', '').split()
                # शब्दों को उनके रेस्पेक्टिव इंडेक्स में बदलें, गायब होने पर <OOV> का उपयोग करें
                seq_idx = [word2idx.get(w, word2idx["<OOV>"]) for w in cleaned_addr]
                text_sequences.append(seq_idx)
            
            # केरास इनपुट के लिए फिक्स्ड लंबाई पर पैडिंग करना
            X_text = pad_sequences(text_sequences, maxlen=self.MAX_LENGTH, padding='post', value=word2idx["<PAD>"])

            # 3. इमेज को बिना लूप ब्लॉक के डायरेक्ट एरे में लोड करना
            X_img_list = []
            for path in df["photo_path"]:
                if os.path.exists(str(path)):
                    img = image.load_img(path, target_size=(64, 64))
                    img_array = image.img_to_array(img) / 255.0 # नॉर्मलाइज़ेशन
                    X_img_list.append(img_array)
                else:
                    logger.warning(f"⚠️ Image not found at {path}. Injecting blank placeholder.")
                    X_img_list.append(np.zeros((64, 64, 3)))
            
            X_img = np.array(X_img_list)
            
            # 4. लेबल्स (Labels/Targets) निकालना
            # यदि CSV में label कॉलम नहीं है, तो डमी बाइनरी एरे बना लें
            y = df["label"].values if "label" in df.columns else np.random.randint(0, 2, size=len(df))

            return X_text, X_img, y, vocab_size, idx2word

        except Exception as e:
            raise CustomException(e, sys)

    def run_training_loop(self):
        try:
            # डेटा और वोकैबुलरी साइज़ प्राप्त करें
            X_text, X_img, y, vocab_size, idx2word = self.load_prepare_data()

            # 1. ModelRegistry से डायनामिक कंबाइंड मॉडल लोड करना
            logger.info("🏗️ Loading CombineCNNRNN via Singleton Model Registry...")
            registry = ModelRegistry(vocab_size=vocab_size, max_length=self.MAX_LENGTH)
            combine_pipeline = registry.get_combinecnnrnn()
            model = combine_pipeline.model # कोर Keras मॉडल एक्सट्रैक्ट करना

            # 2. आर्टिफैक्ट्स फोल्डर और चेकपॉइंट्स सेटअप
            artifacts_dir = os.path.join("backend", "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            saved_model_path = os.path.join(artifacts_dir, "combine_cnn_rnn_model.h5")

            callbacks = [
                EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1),
                ModelCheckpoint(filepath=saved_model_path, monitor='loss', save_best_only=True, verbose=1)
            ]

            # 3. मल्टी-मोडल इनपुट फीडिंग और फिटिंग (Model Fit)
            logger.info("🔥 Commencing Neural Network Training Loop...")
            history = model.fit(
                x={
                    "text_input": X_text, 
                    "image_input": X_img
                },
                y=y,
                epochs=12,
                batch_size=2, # छोटा बैच क्योंकि लिमिटेड डेटासेट है
                callbacks=callbacks,
                verbose=1
            )

            logger.info(f"🎉 Success! Multi-Modal Model saved at: {saved_model_path}")
            return history, idx2word

        except Exception as e:
            raise CustomException(e, sys)

# =====================================================================
# PIPELINE TRIGGER INTERFACE
# =====================================================================
if __name__ == "__main__":
    try:
        trainer_pipeline = CustomModelTrainer()
        trainer_pipeline.run_training_loop()
    except Exception as e:
        logger.error(f"❌ Custom Training Pipeline Aborted: {str(e)}")




import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# आपके प्रोजेक्ट के कस्टम मॉड्यूल्स
from exception.CustomException import CustomException
from logger.logger import get_logger
from all_model_pipeline import ModelRegistry 

logger = get_logger(__name__)

class CompleteMultiVocabularyTrainer:
    def __init__(self):
        logger.info("🎬 Master Training Pipeline for All RNN Topologies Initialized...")
        self.RAW_FILE_PATH = os.path.join("backend", "artifacts", "raw", "raw.csv")
        self.MAX_LEN = 10         # Many-To-X इनपुट के लिए मैक्स लेंथ
        self.MAX_TARGET_LEN = 5  # X-To-Many आउटपुट के लिए मैक्स लेंथ

    def build_custom_vocabulary(self, df):
        """सभी पतों और नामों को प्रोसेस करके कस्टम वोकैब डिक्शनरी बनाना"""
        try:
            logger.info("🔤 Building Master Vocabulary maps...")
            all_words = set(df["name"].astype(str).tolist()) 
            for seq in df["address"].astype(str):
                all_words.update(seq.replace(',', '').split()) 
            
            all_words.add("<PAD>") 
            all_words.add("<OOV>") 

            word2idx = {word: idx for idx, word in enumerate(sorted(all_words))}
            idx2word = {idx: word for word, idx in word2idx.items()}
            vocab_size = len(word2idx)
            
            return word2idx, idx2word, vocab_size
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_data_matrices(self, df, word2idx):
        """चारों मॉडल्स की आर्किटेक्चर के हिसाब से अलग-अलग इनपुट-आउटपुट मैट्रिक्स तैयार करना"""
        try:
            # ए पते (Address) को टोकनाइज़ करना
            address_seqs = []
            for addr in df["address"].astype(str):
                tokens = addr.replace(',', '').split()
                address_seqs.append([word2idx.get(w, word2idx["<OOV>"]) for w in tokens])
            
            # बी नामों (Names) को टोकनाइज़ करना
            name_seqs = []
            for nm in df["name"].astype(str):
                name_seqs.append([word2idx.get(nm, word2idx["<OOV>"])])

            # --- 1. ONE-TO-ONE DATA ---
            X_one_to_one = np.array(name_seqs) # Shape: (samples, 1)
            y_one_to_one = np.array(name_seqs) # Shape: (samples, 1)

            # --- 2. ONE-TO-MANY DATA ---
            X_one_to_many = np.array(name_seqs) # Shape: (samples, 1)
            # आउटपुट को टाइम स्टेप्स के हिसाब से पैड करना होगा
            y_one_to_many = pad_sequences(address_seqs, maxlen=self.MAX_TARGET_LEN, padding='post', value=word2idx["<PAD>"])
            y_one_to_many = np.expand_dims(y_one_to_many, axis=-1) # Shape: (samples, max_target_len, 1)

            # --- 3. MANY-TO-ONE DATA ---
            X_many_to_one = pad_sequences(address_seqs, maxlen=self.MAX_LEN, padding='post', value=word2idx["<PAD>"])
            # आउटपुट के लिए वन-हॉट एनकोडिंग या टारगेट इंडेक्स की आवश्यकता होगी (यहाँ सिंगल आईडी प्रेडिक्शन मान रहे हैं)
            y_many_to_one = np.array([seq[0] for seq in name_seqs]) # Shape: (samples,)

            # --- 4. MANY-TO-MANY DATA ---
            X_many_to_many = pad_sequences(address_seqs, maxlen=self.MAX_LEN, padding='post', value=word2idx["<PAD>"])
            y_many_to_many = pad_sequences(address_seqs, maxlen=self.MAX_TARGET_LEN, padding='post', value=word2idx["<PAD>"])
            # टाइम डिस्ट्रीब्यूटेड लेयर के लिए 3D शेप आवश्यक है
            y_many_to_many = np.expand_dims(y_many_to_many, axis=-1) # Shape: (samples, max_target_len, 1)

            return (X_one_to_one, y_one_to_one, 
                    X_one_to_many, y_one_to_many, 
                    X_many_to_one, y_many_to_one, 
                    X_many_to_many, y_many_to_many)

        except Exception as e:
            raise CustomException(e, sys)

    def fit_all_models(self):
        try:
            # 1. डेटा लोड और वोकैब जनरेशन
            if not os.path.exists(self.RAW_FILE_PATH):
                raise FileNotFoundError(f"Dataset sheets not found at {self.RAW_FILE_PATH}")
            df = pd.read_csv(self.RAW_FILE_PATH)
            
            word2idx, idx2word, vocab_size = self.build_custom_vocabulary(df)
            
            # 2. सभी मैट्रिक्स प्राप्त करें
            (X11, y11, X1M, y1M, XM1, yM1, XMM, yMM) = self.prepare_data_matrices(df, word2idx)

            # 3. सिंगलटन रजिस्ट्री लोड करें
            registry = ModelRegistry(vocab_size=vocab_size, max_length=self.MAX_LEN, max_target_length=self.MAX_TARGET_LEN)


            artifacts_dir = os.path.join("backend", "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)

            early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)

            # =========================================================
            # TRAINING TASK 1: ONE-TO-ONE
            # =========================================================
            logger.info("🔥 Training RNN One-To-One Model...")
            m11 = registry.get_one_to_one_model()
            m11.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            m11.model.fit(X11, y11, epochs=5, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "m11.h5"), save_best_only=True, monitor='loss')], verbose=1)

            # =========================================================
            # TRAINING TASK 2: ONE-TO-MANY
            # =========================================================
            logger.info("🔥 Training RNN One-To-Many Model...")
            m1M = registry.get_one_to_many()
            m1M.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            m1M.model.fit(X1M, y1M, epochs=5, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "m1M.h5"), save_best_only=True, monitor='loss')], verbose=1)

            # =========================================================
            # TRAINING TASK 3: MANY-TO-ONE
            # =========================================================
            logger.info("🔥 Training RNN Many-To-One Model...")
            mM1 = registry.get_many_to_one()
            mM1.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            mM1.model.fit(XM1, yM1, epochs=5, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "mM1.h5"), save_best_only=True, monitor='loss')], verbose=1)

            # =========================================================
            # TRAINING TASK 4: MANY-TO-MANY
            # =========================================================
            logger.info("🔥 Training RNN Many-To-Many Model...")
            mMM = registry.get_many_to_many()
            mMM.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            mMM.model.fit(XMM, yMM, epochs=5, batch_size=2, callbacks=[early_stop, ModelCheckpoint(os.path.join(artifacts_dir, "mMM.h5"), save_best_only=True, monitor='loss')], verbose=1)

            logger.info("🎉 All 4 Recurrent Neural Models trained and saved inside artifacts successfully!")

        except Exception as e:
            raise CustomException(e, sys)

# =====================================================================
# SYSTEM EXECUTION
# =====================================================================
if __name__ == "__main__":
    try:
        master_trainer = CompleteMultiVocabularyTrainer()
        master_trainer.fit_all_models()
    except Exception as e:
        logger.error(f"❌ Master Training Pipeline Interrupted: {str(e)}")
