"""
Main Pipeline Script
Author: Mohammed Ibrahim Kareem
Project: Hybrid Rough Set + CNN + XGBoost Framework
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

print("=== Loading and Preprocessing Data ===")
df = pd.read_csv("../data/final(2).csv")
df = df.rename(columns={'Protcol': 'Protocol'})
cols_to_drop = ['SeddAddress', 'ExpAddress', 'Time', 'IPaddress']
df = df.drop(cols_to_drop, axis=1, errors='ignore')

numeric_cols = ['Clusters', 'BTC', 'USD', 'Netflow_Bytes', 'Port']
categorical_cols = ['Protocol', 'Flag', 'Family', 'Threats']
target_col = 'Prediction'

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col} with {len(le.classes_)} categories")

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[target_col])
class_names = target_encoder.classes_

X_full = df.drop(target_col, axis=1)
X_train_full, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y)

print("\n=== Rough Set Feature Selection ===")
def rough_set_reduction(X_df, y_series, threshold=0.9):
    X = X_df.copy()
    X['target'] = y_series
    features = [col for col in X.columns if col != 'target']
    selected = []
    current_dep = 0
    single_deps = {}
    for f in features:
        dep = X.groupby(f)['target'].apply(lambda x: x.nunique() == 1).mean()
        single_deps[f] = dep
    candidates = sorted(single_deps.items(), key=lambda x: -x[1])
    for f, dep in candidates:
        temp_features = selected + [f]
        current_dep = X.groupby(temp_features)['target'].apply(lambda x: x.nunique() == 1).mean()
        if current_dep > dep or not selected:
            selected.append(f)
            print(f"Added {f}, dependency: {current_dep:.3f}")
            if current_dep >= threshold:
                break
    return selected

selected_features = rough_set_reduction(X_train_full, pd.Series(y_train))
print(f"Selected {len(selected_features)} features: {selected_features}")

def build_cnn_models(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    features = Dense(128, activation='relu', name='deep_features')(x)
    outputs = Dense(n_classes, activation='softmax')(features)
    train_model = Model(inputs=inputs, outputs=outputs)
    train_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    feature_model = Model(inputs=inputs, outputs=features)
    return train_model, feature_model

X_train_selected = X_train_full[selected_features]
X_test_selected = X_test[selected_features]
X_train_cnn = np.expand_dims(X_train_selected.values, axis=2)
X_test_cnn = np.expand_dims(X_test_selected.values, axis=2)

print("\n=== CNN Training ===")
train_model, feature_model = build_cnn_models((X_train_cnn.shape[1], 1), len(class_names))
history = train_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test), epochs=20, batch_size=64, verbose=1)

print("\n=== Feature Extraction ===")
train_features = feature_model.predict(X_train_cnn)
test_features = feature_model.predict(X_test_cnn)

print("\n=== XGBoost Training ===")
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_names),
    n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
xgb_model.fit(train_features, y_train)

def evaluate_model(model, features, labels, set_name):
    preds = model.predict(features)
    print(f"\n=== {set_name} Set Evaluation ===")
    print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
    print(classification_report(labels, preds, target_names=class_names))
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{set_name} Set Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

evaluate_model(xgb_model, train_features, y_train, "Training")
evaluate_model(xgb_model, test_features, y_test, "Test")

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

print("\n=== Saving Models ===")
joblib.dump({
    'selected_features': selected_features,
    'train_model': train_model,
    'feature_model': feature_model,
    'xgb_model': xgb_model,
    'target_encoder': target_encoder,
    'feature_encoders': label_encoders,
    'scaler': scaler,
    'class_names': class_names,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols
}, '../models/hybrid_model_pipeline.pkl')
print("Pipeline Saved Successfully.")
