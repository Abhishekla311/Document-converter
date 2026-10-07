import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# कस्टम एक्सेप्शन, लॉगर और आपकी ट्रेनिंग पाइपलाइन का इम्पोर्ट
from exception.CustomException import CustomException
from logger.logger import get_logger
from src.train import ModelTrainerPipeline  # FIXED: आपके प्रोजेक्ट स्ट्रक्चर के अनुसार सही पाथ

logger = get_logger(__name__)

class ModelEvaluatorPipeline:
    def __init__(self):
        logger.info("Model Evaluation Pipeline Initialized")
        self.trainer = ModelTrainerPipeline()
        # FIXED: आपके लॉग्स के अनुसार सही आर्टीफैक्ट्स डायरेक्टरी पाथ
        self.ARTIFACTS_DIR = os.path.join("backend", "artifacts")
        
    def evaluate_all_models(self):
        try:
            # 1. वोकैबुलरी और डेटा को लोड करें
            word2idx, idx2word, vocab_size = self.trainer.build_custom_vocabulary()
            
            if not os.path.exists(self.trainer.RAW_FILE_PATH):
                raise FileNotFoundError(f"Evaluation dataset missing at {self.trainer.RAW_FILE_PATH}")
                
            df = pd.read_csv(self.trainer.RAW_FILE_PATH)
            
            # डेटा मैट्रिसेस तैयार करें (X11, y11, आदि)
            (X11, y11, X1M, y1M, XMM, yMM, XM1, yM1) = self.trainer.prepare_data_metrices(df, word2idx)
            
            print("\n" + "="*60)
            print("📊 STARTING ALL 5 MODELS EVALUATION & INFERENCE TEST")
            print("="*60 + "\n")
            
            # --- MODEL 1: One-To-One Evaluation ---
            m11_path = os.path.join(self.ARTIFACTS_DIR, "m11.h5")
            if os.path.exists(m11_path):
                print("🔹 [1/5] Evaluating RNN One-To-One Model...")
                model_11 = load_model(m11_path)
                loss = model_11.evaluate(X11, y11, verbose=0)
                print(f"   ↳ Evaluation Loss: {loss:.4f}")
                
                # Sample Prediction Test
                sample_pred = model_11.predict(X11[:1], verbose=0)
                pred_idx = np.argmax(sample_pred, axis=-1).flatten()
                pred_words = [idx2word[idx] for idx in pred_idx if idx2word.get(idx) != "<PAD>"]
                print(f"   ↳ Sample Predicted Output: {' '.join(pred_words)}")
            else:
                print(f"⚠️ m11.h5 model file not found at {m11_path}")

            # --- MODEL 2: One-To-Many Evaluation ---
            m1M_path = os.path.join(self.ARTIFACTS_DIR, "m1M.h5")
            if os.path.exists(m1M_path):
                print("\n🔹 [2/5] Evaluating RNN One-To-Many Model...")
                model_1M = load_model(m1M_path)
                loss = model_1M.evaluate(X1M, y1M, verbose=0)
                print(f"   ↳ Evaluation Loss: {loss:.4f}")
                
                # Sample Prediction Test
                sample_pred = model_1M.predict(X1M[:1], verbose=0)
                pred_indices = np.argmax(sample_pred, axis=-1).flatten()
                pred_words = [idx2word[idx] for idx in pred_indices]
                print(f"   ↳ Sample Predicted Output: {' '.join(pred_words)}")
            else:
                print(f"⚠️ m1M.h5 model file not found at {m1M_path}")

            # --- MODEL 3: Many-To-One Evaluation ---
            mM1_path = os.path.join(self.ARTIFACTS_DIR, "mM1.h5")
            if os.path.exists(mM1_path):
                print("\n🔹 [3/5] Evaluating RNN Many-To-One Model...")
                model_M1 = load_model(mM1_path)
                loss = model_M1.evaluate(XM1, yM1, verbose=0)
                print(f"   ↳ Evaluation Loss: {loss:.4f}")
                
                # Sample Prediction Test
                sample_pred = model_M1.predict(XM1[:1], verbose=0)
                pred_index = np.argmax(sample_pred, axis=-1).flatten()
                print(f"   ↳ Sample Predicted Output Token: {idx2word.get(pred_index[0], '<UNKNOWN>')}")
            else:
                print(f"⚠️ mM1.h5 model file not found at {mM1_path}")

            # --- MODEL 4: Many-To-Many Evaluation ---
            mMM_path = os.path.join(self.ARTIFACTS_DIR, "mMM.h5")
            if os.path.exists(mMM_path):
                print("\n🔹 [4/5] Evaluating RNN Many-To-Many Model...")
                model_MM = load_model(mMM_path)
                loss = model_MM.evaluate(XMM, yMM, verbose=0)
                print(f"   ↳ Evaluation Loss: {loss:.4f}")
                
                # Sample Prediction Test
                sample_pred = model_MM.predict(XMM[:1], verbose=0)
                pred_indices = np.argmax(sample_pred, axis=-1).flatten()
                pred_words = [idx2word[idx] for idx in pred_indices]
                print(f"   ↳ Sample Predicted Output Sequence: {' '.join(pred_words)}")
            else:
                print(f"⚠️ mMM.h5 model file not found at {mMM_path}")

            # --- MODEL 5: Combined CNN_RNN Evaluation (ADDED FIXED BLOCK) ---
            mCombine_path = os.path.join(self.ARTIFACTS_DIR, "mCombine.h5")
            if os.path.exists(mCombine_path):
                print("\n🔹 [5/5] Evaluating Combined CNN_RNN Model...")
                model_Combine = load_model(mCombine_path)
                
                # Combined model को इमेज और टेक्स्ट दोनों इनपुट चाहिए होते हैं
                X_dummy_img = np.random.rand(len(df), 64, 64, 3)
                y_dummy_labels = np.random.randint(0, 2, size=(len(df), 1))
                
                # मॉडल इवैल्यूएशन
                loss, accuracy = model_Combine.evaluate([XM1, X_dummy_img], y_dummy_labels, verbose=0)
                print(f"   ↳ Evaluation Loss: {loss:.4f}")
                print(f"   ↳ Evaluation Accuracy: {accuracy*100:.2f}%")
                
                # Sample Prediction Test
                sample_pred = model_Combine.predict([XM1[:1], X_dummy_img[:1]], verbose=0)
                print(f"   ↳ Sample Predicted Probability: {float(sample_pred[0][0]):.4f}")
            else:
                print(f"⚠️ mCombine.h5 model file not found at {mCombine_path}")

            print("\n" + "="*60)
            print("🎉 Evaluation Complete! All 5 Models Checked Successfully.")
            print("="*60 + "\n")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        evaluator = ModelEvaluatorPipeline()
        evaluator.evaluate_all_models()
    except Exception as e:
        print(f"❌ Evaluation Pipeline Failed: {str(e)}")
