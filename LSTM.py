import numpy as np
import tensorflow as tf
from tensorflow import keras
import pyaudio
import librosa
# import noisereduce as nr
import os
import time
from pydub import AudioSegment, effects

# Load the model
model = keras.models.load_model('Results/LSTM_Raw/13/best_weights.h5')

# Define the list of emotions
emotions = ['happy', 'sad', 'angry', 'neutral']

# Define preprocessing function
def preprocess(audio, sr):
    # Normalize the audio to +5.0 dBFS.
    normalizedsound = audio.apply_gain(5 - audio.dBFS)

    # Transform the normalized audio to np.array of samples.
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')

    # Trim silence from the beginning and the end.
    xt, index = librosa.effects.trim(normal_x, top_db=30)

    # Pad for duration equalization.
    total_length = 2 * 300000  # 4 seconds
    padded_x = np.pad(xt, (0, total_length - len(xt)), 'constant')

    # Noise reduction.
#     final_x = nr.reduce_noise(padded_x, sr=sr)

    # Features extraction
    frame_length = 2048
    hop_length = 512
    f1 = librosa.feature.rms(padded_x, frame_length=frame_length, hop_length=hop_length) # Energy - Root Mean Square   
    f2 = librosa.feature.zero_crossing_rate(padded_x , frame_length=frame_length, hop_length=hop_length, center=True) # ZCR      
    f3 = librosa.feature.mfcc(padded_x, sr=sr, n_mfcc=13, hop_length = hop_length) # MFCC

    # Stack features
    features = np.vstack([f1, f2, f3])

    # The LSTM model likely expects a 3-dimensional input, where the dimensions correspond to
    # (number of examples, number of time steps, number of features).
    # Add an extra dimension to the start of the shape to represent a single example.
    features = np.expand_dims(features, axis=0)

    return features

def pad_sequence(seq, max_length):
    num_padding = max_length - seq.shape[1]
    if num_padding > 0:
        return np.pad(seq, ((0, 0), (0, num_padding), (0, 0)), 'constant')
    else:
        return seq

# Initialize PyAudio
pa = pyaudio.PyAudio()
stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=1024)

# Calculate the number of buffers needed for 4 seconds of audio
buffers_per_second = 44100 // 1024
buffers_per_four_seconds = 4 * buffers_per_second

time.sleep(2) # wait for 2 seconds before starting

print("Speak now: ")
# Capture 4 seconds of audio
buffer = b""
for _ in range(buffers_per_four_seconds):
    buffer += stream.read(1024)

# Convert buffer to AudioSegment
audio_data = AudioSegment(buffer, sample_width=2, frame_rate=44100, channels=1)

# Preprocess audio to create features
features = preprocess(audio_data, sr=44100)

# Transpose features
features = np.transpose(features, (0, 2, 1))

# Predict emotion
prediction = model.predict(features)

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