import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Dense

# Create the Flask app
app = Flask(__name__)

# Define the ConformerBlock class
class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, ff_dim, dropout_rate=0.1):
        super(ConformerBlock, self).__init__()
        self.conv1 = Conv2D(filters=dim, kernel_size=(1, 3), padding='same', activation='relu')
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att = tfa.layers.MultiHeadAttention(head_size=dim, num_heads=num_heads)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(dim)
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

# Load the trained model with custom object scope
with tf.keras.utils.custom_object_scope({'ConformerBlock': ConformerBlock}):
    model = load_model('11eeg_classification_model_conformer.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Simulation parameters
num_channels = 8
num_time_points = 500
sampling_rate = 250

# Define the words and associated frequency bands
words = ["RogerThat", "AlphaCharlie", "LimaCharlie", "OscarMike", "CharlieMike", "BravoZulu", "Papa", "Alpha"]

frequency_bands = {
    "RogerThat": (1, 3),
    "AlphaCharlie": (4, 6),
    "LimaCharlie": (7, 9),
    "OscarMike": (10, 12),
    "CharlieMike": (13, 15),
    "BravoZulu": (16, 18),
    "Papa": (19, 21),
    "Alpha": (22, 24)
}

# Function to simulate EEG signal for a word
def simulate_eeg_signal(word, num_channels, num_time_points, sampling_rate):
    t = np.linspace(0, num_time_points / sampling_rate, num_time_points)
    signal = np.zeros((num_channels, num_time_points))

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

# Function to preprocess EEG data
def preprocess_data(eeg_signal):
    eeg_signal = np.array(eeg_signal).reshape(1, num_channels, num_time_points, 1)
    return eeg_signal

# Home route to display the buttons
@app.route('/')
def index():
    return render_template('index.html', words=words)

# Route to handle button click and prediction
@app.route('/predict/<word>')
def predict(word):
    # Simulate EEG signal for the word
    eeg_signal = simulate_eeg_signal(word, num_channels, num_time_points, sampling_rate)
    preprocessed_signal = preprocess_data(eeg_signal)

    # Make prediction
    prediction = model.predict(preprocessed_signal)
    predicted_label_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

    return render_template('result.html', word=word, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
