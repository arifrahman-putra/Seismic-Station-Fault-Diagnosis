import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import joblib
import json
import os

# Load datasets
Dataset_path = "C:\\Data\\Publikasi Ilmiah\\Predictive Maintenance\\Codes\\Github Upload\\Dataset_ALL_NEW_Deviation.xlsx"
df_all = pd.read_excel(Dataset_path)

# normal dataframe for One_AE anomaly detection
df_normal = df_all[df_all["HS"] == 1]
X_normal = df_normal[["H_4","H_19","H_100","H_325"]].values

# abnormal dataframe for DNN classifier
df_abnormal = df_all[df_all["HS"] == -1]
X_abnormal = df_abnormal[["H_4","H_19","H_100","H_325"]].values
y_abnormal = df_abnormal["Fault"].values

le = LabelEncoder()
y_abnormal_encoded = le.fit_transform(y_abnormal)

# Scalers
One_AE_scaler = MinMaxScaler()
X_train_One_AE = One_AE_scaler.fit_transform(X_normal)

DNN_scaler = StandardScaler()
X_train_DNN = DNN_scaler.fit_transform(X_abnormal)


# Autoencoder model
input_layer = Input(shape=(X_train_One_AE.shape[1],))
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(X_train_One_AE.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

# Train the autoencoder
One_AE_history = autoencoder.fit(X_train_One_AE, X_train_One_AE, epochs=200, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)

# Calculate threshold
X_train_recon = autoencoder.predict(X_train_One_AE)
recon_error_train = np.mean(np.square(X_train_One_AE - X_train_recon), axis=1)
threshold = np.percentile(recon_error_train, 95)
print(f"One_AE threshold: {threshold}")
print(f"\nAnomaly detection threshold: {threshold}")


# DNN Classifier model
input_layer = Input(shape=(X_train_DNN.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(len(le.classes_), activation='softmax')(x)
print("number of fault classes: ", le.classes_)

DNN = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
DNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
DNN_history = DNN.fit(X_train_DNN, y_abnormal_encoded, epochs=400, batch_size=32, validation_split=0.2)


# ========== SAVE ALL COMPONENTS ==========
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# 1. Save Autoencoder model
autoencoder.save(f"{save_dir}/autoencoder_model.h5")
print("\n✓ Autoencoder model saved")

# 2. Save DNN Classifier model
DNN.save(f"{save_dir}/dnn_classifier_model.h5")
print("✓ DNN classifier model saved")

# 3. Save scalers
joblib.dump(One_AE_scaler, f"{save_dir}/one_ae_scaler.pkl")
joblib.dump(DNN_scaler, f"{save_dir}/dnn_scaler.pkl")
print("✓ Scalers saved")

# 4. Save label encoder
joblib.dump(le, f"{save_dir}/label_encoder.pkl")
print("✓ Label encoder saved")

# 5. Save threshold and metadata
metadata = {
    'threshold': float(threshold),
    'feature_names': ["H_4", "H_19", "H_100", "H_325"],
    'fault_classes': le.classes_.tolist(),
    'num_classes': len(le.classes_)
}


with open(f"{save_dir}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=4)
print("✓ Metadata saved")

print(f"\n✓✓✓ All models saved to: {save_dir}/")