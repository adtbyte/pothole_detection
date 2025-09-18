import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ----------------- Load environment -----------------
load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH", r"path for dataset")
TRAINING_DIR = os.path.join(DATASET_PATH, "train")
TESTING_DIR = os.path.join(DATASET_PATH, "test")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "pothole_cnn_model_v3.keras")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# ----------------- Data Generators -----------------
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', classes=['Plain', 'Pothole']
    )
    validation_generator = validation_datagen.flow_from_directory(
        TESTING_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', classes=['Plain', 'Pothole'], shuffle=False
    )
    return train_generator, validation_generator

# ----------------- Build Model -----------------
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), padding='same'),
        BatchNormalization(), MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ----------------- Training -----------------
def run_training():
    st.info("Starting model training process...")

    if not os.path.isdir(TRAINING_DIR) or not os.path.isdir(TESTING_DIR):
        st.error(f"Error: Dataset path is invalid. Make sure '{TRAINING_DIR}' and '{TESTING_DIR}' exist.")
        return

    train_generator, validation_generator = create_data_generators()
    if train_generator.n == 0 or validation_generator.n == 0:
        st.error("Training failed: No images found. Please check your dataset folders.")
        return

    st.write(f"Found {train_generator.n} images for training.")
    st.write(f"Found {validation_generator.n} images for validation.")

    st.write("Calculating class weights to handle data imbalance...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    st.write(f"Weights calculated: Class 0 (Plain) = {class_weights[0]:.2f}, Class 1 (Pothole) = {class_weights[1]:.2f}")

    st.write("Building and training the model with class weights...")
    model = build_model()
    model.summary(print_fn=lambda x: st.text(x))

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict
    )

    st.success("✅ Model training complete!")
    best_model = load_model(MODEL_PATH)

    y_pred_probs = best_model.predict(validation_generator)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = validation_generator.classes

    report = classification_report(
        y_true, y_pred, target_names=['Plain', 'Pothole'],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    st.write("### Model Evaluation Report (on validation data)")
    st.dataframe(pd.DataFrame(report).transpose())
    st.write("### Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=['True Plain', 'True Pothole'], columns=['Predicted Plain', 'Predicted Pothole']))
    st.success(f"Best model saved successfully to: `{MODEL_PATH}`")

# ----------------- Pothole Detection -----------------
def detect_and_draw_pothole(image):
    output_image = image.copy()
    height = image.shape[0]
    roi_start_row = int(height * 0.4)
    road_roi = image[roi_start_row:, :]
    gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pothole_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 50000:
            pothole_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            original_x = x
            original_y = y + roi_start_row
            cv2.rectangle(output_image, (original_x, original_y),
                           (original_x + w, original_y + h), (0, 0, 255), 2)
            cv2.putText(output_image, "Pothole", (original_x, original_y - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if pothole_count > 0:
        cv2.putText(output_image, f"Potholes Detected: {pothole_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_image, pothole_count

# ----------------- Sustainability Impact -----------------
def estimate_co2_savings(pothole_count):
    # Simple assumption: fixing one pothole saves ~5kg CO₂/year per 1000 vehicles
    co2_saved = pothole_count * 5
    return co2_saved

# ----------------- Streamlit App -----------------
st.set_page_config(layout="wide")
st.title("🚧 Pothole Detector using CNN + 🌱 Green AI Sustainability")
st.markdown("This app uses an enhanced **CNN** to classify and locate potholes in road images, while also estimating **CO₂ savings** from sustainable road repairs.")

# Sidebar - Training
st.sidebar.header("⚙️ Model Training")
st.sidebar.info("Your dataset should have the structure shown in the app.")
if st.sidebar.button("Train / Re-train Model"):
    with st.spinner('Training in progress... This may take some time.'):
        run_training()

# Sidebar - Model Status
st.sidebar.header("ℹ️ Model Status")
if os.path.exists(MODEL_PATH):
    st.sidebar.success(f"✔️ Model found: `{MODEL_PATH}`")
else:
    st.sidebar.warning("Model not found. Please train a model.")

# Prediction Section
st.header("🔍 Predict on a New Image")
uploaded_file = st.file_uploader("Upload a road image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if not os.path.exists(MODEL_PATH):
        st.error("Cannot predict: Model has not been trained yet.")
    else:
        model = load_model(MODEL_PATH)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Analyzing image..."):
                img_resized = cv2.resize(image_rgb, IMG_SIZE)
                img_array = np.expand_dims(img_resized, axis=0) / 255.0
                prediction_prob = float(model.predict(img_array)[0][0])

                st.write("### 📊 Prediction Result")

                HIGH_CONF_THRESHOLD = 0.75
                if prediction_prob > HIGH_CONF_THRESHOLD:
                    confidence = prediction_prob * 100
                    st.error(f"**Result: Pothole Detected** (Confidence: {confidence:.2f}%)")
                    st.write("---")
                    st.write("#### Pothole Localization:")
                    annotated_image, pothole_count = detect_and_draw_pothole(image_bgr.copy())
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed Image")

                    # Sustainability Impact
                    co2_saved = estimate_co2_savings(pothole_count)
                    st.metric("🌍 Potential CO₂ Savings", f"{co2_saved} kg/year")

                    if st.button("Export Sustainability Report"):
                        pothole_data = pd.DataFrame({
                            "Potholes Detected": [pothole_count],
                            "Estimated CO₂ Savings (kg/year)": [co2_saved]
                        })
                        pothole_data.to_csv("sustainability_report.csv", index=False)
                        st.success("Report generated: sustainability_report.csv")

                elif prediction_prob > 0.5:
                    confidence = prediction_prob * 100
                    st.warning(f"**Result: Possible Pothole** (Confidence: {confidence:.2f}%)")
                    st.info("The model has low confidence. Please verify manually.")
                    annotated_image, pothole_count = detect_and_draw_pothole(image_bgr.copy())
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed Image")

                    co2_saved = estimate_co2_savings(pothole_count)
                    st.metric("🌍 Potential CO₂ Savings", f"{co2_saved} kg/year")

                else:
                    confidence = (1 - prediction_prob) * 100
                    st.success(f"**Result: Plain Road** (Confidence: {confidence:.2f}%)")

                st.write("---")
                st.write("Confidence Scores:")
                st.progress(prediction_prob)
                st.write(f"- Plain Road: `{(1 - prediction_prob) * 100:.2f}%`")
                st.write(f"- Pothole: `{prediction_prob * 100:.2f}%`")
