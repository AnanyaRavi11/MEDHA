import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import tensorflow_addons as tfa
import pickle

# Define the ConformerBlock class
class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, ff_dim, dropout_rate=0.1):
        super(ConformerBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=dim, kernel_size=(1, 3), padding='same', activation='relu')
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att = tfa.layers.MultiHeadAttention(head_size=dim, num_heads=num_heads)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(dim)
        ])
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.att([x, x, x])
        x = self.norm2(x)
        x = self.dropout(x, training=training)
        x = x + inputs
        x = self.ffn(x)
        x = self.norm3(x)
        return x

# Load the model with the custom object scope
with custom_object_scope({'ConformerBlock': ConformerBlock}):
    model = load_model('11eeg_classification_model_conformer.h5', compile=False)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to preprocess EEG data
def preprocess_eeg(eeg_data):
    eeg_data = np.array(eeg_data).reshape(1, 8, 500, 1)  # Reshape to match the model input
    return eeg_data

# Simulate EEG signal for a word
def simulate_eeg_signal(word, num_channels, num_time_points, sampling_rate):
    t = np.linspace(0, num_time_points / sampling_rate, num_time_points)
    signal = np.zeros((num_channels, num_time_points))

    frequency_bands = {
        "Hello": (1, 3),
        "World": (4, 6),
        "Python": (7, 9),
        "Data": (10, 12),
        "Science": (13, 15),
        "Machine": (16, 18),
        "Learning": (19, 21),
        "Model": (22, 24),
        "Brain": (25, 27),
        "Wave": (28, 30)
    }

    band = frequency_bands[word]
    base_freq = np.mean(band)

    for channel in range(num_channels):
        frequency = base_freq  # Fixed frequency for this channel
        amplitude = 1.0  # Consistent amplitude
        noise = np.random.normal(0, 0.02, num_time_points)  # Reduced noise

        # Generate a signal with slight frequency and amplitude variations
        signal[channel] = amplitude * np.sin(2 * np.pi * frequency * t) + noise

        # Adding slight variations in different channels
        if channel % 2 == 0:
            signal[channel] += 0.05 * np.sin(2 * np.pi * (frequency + 0.5) * t)
        else:
            signal[channel] += 0.05 * np.cos(2 * np.pi * (frequency - 0.5) * t)

    return signal

# Simulation parameters
num_channels = 8
num_time_points = 500
sampling_rate = 250  # Hz

st.title("EEG to Text Prediction")

# List to store actual labels and their corresponding predictions
actual_labels = []
predicted_labels = []

# Display predictions for each uploaded image
for i in range(10):
    uploaded_image = st.file_uploader(f"Upload Image {i+1}", type=["png", "jpg", "jpeg"], key=i)
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Extract the label from the image filename
        filename = uploaded_image.name
        actual_label = filename.split('_')[0]
        
        # Simulate EEG signal for the extracted label
        new_eeg_signal = simulate_eeg_signal(actual_label, num_channels, num_time_points, sampling_rate)
        preprocessed_signal = preprocess_eeg(new_eeg_signal)
        
        # Make prediction
        prediction = model.predict(preprocessed_signal)
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
        
        # Store the actual and predicted labels
        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)
        
        # Display the image and labels
        st.image(image, caption=f"Uploaded Image {i+1}", use_column_width=True)
        st.write(f"1/1 [==============================] - 0s")
        st.write(f"Actual Label: {actual_label}")
        st.write(f"Predicted Label: {predicted_label}")
        st.write("------------------------------")

if len([file for file in st.session_state if file.startswith('Upload Image')]) < 10:
    st.write("Please upload 10 images.")
