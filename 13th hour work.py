import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
from scipy.signal import butter, sosfilt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import scipy

# Define the function to extract MFCC
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        if audio is None:
            print("Error: Invalid audio -", file_name)
            return None


        # Apply high-pass filter to the audio
        # cutoff_freq = 1000  # Adjust the cutoff frequency as needed
        # filtered_audio = high_pass_filter(audio, sample_rate, cutoff_freq)
        # audio, _ = librosa.effects.trim(audio)

        # Trim the audio based on a threshold
        threshold = 30  # Adjust the threshold as needed
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=threshold)

        mfccs = librosa.feature.mfcc(y=trimmed_audio, sr=sample_rate, n_mfcc=39)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        return mfccs_processed

    except Exception as e:
        print("Error encountered while parsing file:", file_name)
        print("Exception:", str(e))
        return None

def high_pass_filter(audio, sr, cutoff_freq=1000):
    nyquist_freq = 0.5 * sr
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = scipy.signal.butter(4, normalized_cutoff, btype='high', analog=False)
    filtered_audio = scipy.signal.lfilter(b, a, audio)
    return filtered_audio

audio_dir = 'newData2/'

audio_files = []
for subdir, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav'):
            filepath = os.path.join(subdir, file).replace("\\", "/")
            audio_files.append(filepath)

mfcc_data = []
emotions = []
for i, file in enumerate(audio_files):
    print(f'Processing file {i+1}/{len(audio_files)}: {file}')
    mfccs = extract_features(file)
    if mfccs is not None:
        mfcc_data.append(mfccs)

        # Extracting the emotion from the file name, as per the dataset structure
        file_name = os.path.basename(file)
        emotion = int(file_name.split("-")[1]) - 1  # Subtracting 1 to make the emotions start from 0
        emotions.append(emotion)

num_rows = len(mfcc_data)
num_cols = len(mfcc_data[0])
print("Number of rows:", num_rows)
print("Number of columns:", num_cols)

mfcc_data = np.asarray(mfcc_data)  # Convert mfcc_data to a NumPy array

print("Shape of mfcc_data:", mfcc_data.shape)
data = pd.DataFrame(mfcc_data, columns=[f'mfcc_{i}' for i in range(len(mfcc_data[0]))])
data['emotions'] = emotions

X = data.iloc[:, :-1]
y = data['emotions']

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pickle.dump(scaler, open('Results/SVM/complete dataset/scaler.pkl', 'wb'))

param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'svc__kernel': ['rbf']}

pipe = Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC(probability=True))])

# Perform k-fold cross-validation with GridSearchCV
grid = GridSearchCV(pipe, param_grid, refit=True, verbose=2, cv=5)

# Split the data into training and testing sets with 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)

# Perform grid search only on the training set
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)


pickle.dump(grid, open('Results/SVM/complete dataset/SVM.pkl', 'wb'))
# Evaluate the best estimator on the testing set
y_pred = grid.predict(X_test)

# Calculate test accuracy, classification report, and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")

report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{report}")

confusion = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{confusion}")
