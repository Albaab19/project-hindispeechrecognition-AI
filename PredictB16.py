import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyaudio
import librosa
import numpy as np
from scipy import signal
import noisereduce as nr
from matplotlib import pyplot as plt
import librosa.display
import os
import time
from scipy import ndimage
from keras import models
from keras.models import load_model

# Load the VGGNet model
model = load_model('C:/Users/admin/Results/VGG_Raw/1/modelVGG_B16.h5')
emotions = ['happy', 'sad', 'angry', 'neutral']

# Define preprocessing function
def preprocess(audio, sr):
    # High-pass filter
    b, a = signal.butter(4, 100, 'high', fs=sr)
    audio_hp = signal.filtfilt(b, a, audio)

    # Trim audio
    audio, _ = librosa.effects.trim(audio_hp, top_db=30, frame_length=2048, hop_length=512)

    # Noise reduction
    audio = nr.reduce_noise(audio, sr=sr)

    # Create spectrogram
    spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128)

    # Convert to decibels
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Resize spectrogram to (224, 224)
    spectrogram = ndimage.zoom(spectrogram, (224 / spectrogram.shape[0], 224 / spectrogram.shape[1]))

    # The model expects a 3-channel input, but spectrograms are 1-channel.
    # So, we duplicate the spectrogram across 3 channels to create a 3-channel "image".
    spectrogram = np.stack([spectrogram, spectrogram, spectrogram], axis=-1)

    return spectrogram

# Initialize PyAudio
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Calculate the number of buffers needed for 5 seconds of audio
buffers_per_second = 44100 // 1024
buffers_per_five_seconds = 5 * buffers_per_second


time.sleep(1) 
print("start :")
# Capture 5 seconds of audio
buffer = b""
for _ in range(buffers_per_five_seconds):
    buffer += stream.read(1024)

audio_data = np.frombuffer(buffer, dtype=np.float32)

# Preprocess audio to create spectrogram
spectrogram = preprocess(audio_data, sr=44100)

# Predict emotion
prediction = model.predict(np.array([spectrogram]))

# Get the index of the highest output value
predicted_index = np.argmax(prediction[0])

# Get the name of the emotion
predicted_emotion = emotions[predicted_index]

# Output emotion
print('Predicted emotion:', predicted_emotion)

# Close the stream
stream.stop_stream()
stream.close()
pa.terminate()