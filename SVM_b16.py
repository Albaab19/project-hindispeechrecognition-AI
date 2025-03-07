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

# Define the function to extract MFCC
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        if audio is None:
            print("Error: Invalid audio -", file_name)
            return None

        audio = high_pass_filter(audio, sample_rate)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=39)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        return mfccs_processed

    except Exception as e:
        print("Error encountered while parsing file:", file_name)
        print("Exception:", str(e))
        return None

def high_pass_filter(signal, sr, cutoff=100):
    sos = butter(10, cutoff, 'hp', fs=sr, output='sos')
    filtered = sosfilt(sos, signal)
    return filtered

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
grid.fit(X_scaled, y)

print(grid.best_params_)
print(grid.best_estimator_)

pickle.dump(grid, open('Results/SVM/complete dataset/SVM.pkl', 'wb'))

# Calculate cross-validated test accuracy
cv_accuracy = cross_val_score(grid.best_estimator_, X_scaled, y, cv=5)
print("Cross-Validated Test Accuracy:", np.mean(cv_accuracy))

# Define the test audio directory
test_audio_dir = 'testSVM/'

test_audio_files = os.listdir(test_audio_dir)
test_mfcc_data = []
test_emotions = []

for i, file in enumerate(test_audio_files):
    file_path = os.path.join(test_audio_dir, file)
    print(f'Processing test file {i+1}/{len(test_audio_files)}: {file_path}')
    mfccs = extract_features(file_path)
    if mfccs is not None:
        test_mfcc_data.append(mfccs)

        # Extract the emotion from the file name
        emotion = int(file.split("-")[1]) - 1
        test_emotions.append(emotion)

test_data = np.asarray(test_mfcc_data)

if len(test_data) > 0:
    test_X = pd.DataFrame(test_data, columns=[f'mfcc_{i}' for i in range(len(test_data[0]))])
    test_X_scaled = scaler.transform(test_X)

    # Make predictions on the test set
    y_pred = grid.predict(test_X_scaled)

    # Calculate the accuracy
    accuracy = accuracy_score(test_emotions, y_pred)
    print("Test Accuracy:", accuracy)

    # Generate the classification report
    classification_report = classification_report(test_emotions, y_pred)
    print("Classification Report:")
    print(classification_report)

    # Generate the confusion matrix
    confusion_matrix = confusion_matrix(test_emotions, y_pred)
    print("Confusion Matrix:")
    print(confusion_matrix)
else:
    print("No valid test audio files found in the specified directory.")

