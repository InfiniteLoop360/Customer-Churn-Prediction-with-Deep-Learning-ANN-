# ===============================
# 1. Import Libraries
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# Fix random seeds
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# # ===============================
# # OPTION A: Upload CSV manually
# # ===============================
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('/content/Churn_Modelling (1).csv')
df.head()

# ===============================
# OPTION B: Load from Google Drive
# ===============================
# from google.colab import drive
# drive.mount('/content/drive')

# df = pd.read_csv('/content/Churn_Modelling (1).csv')
# df.head()


# ===============================
# 3. Data Cleaning & Preprocessing
# ===============================

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

# Check missing values
print("Missing Values:")
print(df.isna().sum())

# One-hot encoding
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
df.head()


# ===============================
# 4. Feature Split + Scaling
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=['Exited'])
y = df['Exited'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

X_train.shape, X_test.shape


# ===============================
# 5. Handle Class Imbalance
# ===============================

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
class_weight_dict


# ===============================
# 6. Model Building
# ===============================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

n_features = X_train_trf.shape[1]

model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()


# ===============================
# 7. Train the Model
# ===============================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    X_train_trf, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=2
)


# ===============================
# 8. Model Evaluation
# ===============================

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

y_prob = model.predict(X_test_trf).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 9. Plot Training Curves
# ===============================

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Validation'])

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['Train', 'Validation'])

plt.show()


# ===============================
# 10. Save Model and Scaler
# ===============================

model.save('/content/churn_model.h5')

import joblib
joblib.dump(scaler, '/content/scaler.save')

print("Files saved successfully!")


from google.colab import files
files.download('/content/churn_model.h5')
files.download('/content/scaler.save')

