import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import json


class PMInference:
    def __init__(self, model_dir="saved_models"):
        self.autoencoder = load_model(f"{model_dir}/autoencoder_model.h5")
        self.dnn_classifier = load_model(f"{model_dir}/dnn_classifier_model.h5")
        self.one_ae_scaler = joblib.load(f"{model_dir}/one_ae_scaler.pkl")
        self.dnn_scaler = joblib.load(f"{model_dir}/dnn_scaler.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")

        with open(f"{model_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        self.threshold = metadata['threshold']
        print(f"Models loaded. Threshold: {self.threshold}")

    def predict(self, X):
        """
        Input: X array (n_samples, 4) - [H_4, H_19, H_100, H_325]
        Output: array of predictions ('Normal' or fault class name)
        """
        # Anomaly detection
        X_scaled_ae = self.one_ae_scaler.transform(X)
        X_recon = self.autoencoder.predict(X_scaled_ae, verbose=0)
        recon_error = np.mean(np.square(X_scaled_ae - X_recon), axis=1)
        is_anomaly = (recon_error > self.threshold).astype(int)

        # Fault classification
        X_scaled_dnn = self.dnn_scaler.transform(X)
        fault_probs = self.dnn_classifier.predict(X_scaled_dnn, verbose=0)
        fault_classes = self.label_encoder.inverse_transform(np.argmax(fault_probs, axis=1))

        # Combine: Normal jika tidak anomaly, fault class jika anomaly
        predictions = np.where(is_anomaly == 0, "Normal", fault_classes)
        return predictions


# Contoh penggunaan
if __name__ == "__main__":
    pm = PMInference()

    # Contoh data
    test_data = np.array([[0.5, 0.3, 0.2, 0.1]])
    result = pm.predict(test_data)
    print(f"Prediction: {result}")