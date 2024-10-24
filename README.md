step-1 : Install the below librearies using following below commands

pip install sounddevice #for sound
pip install wave #analyse waves
pip install numpy #for computational
pip install tkinter #UI
pip install customtkinter #UI
pip install tensorflow #for dll models
pip install scikit-learn 
pip install librosa #extract audio features
pip install pandas #for computational
pip install pyinstaller #for software conversion

step-2 : dataset and its features

Dataset link (RAVDESS) : https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view?pli=1
Extraction of features from audio  : dataset.csv 

Features Involved
1. Temporal Features
  •	Zero-Crossing Rate: The rate at which the signal changes sign (crosses the zero amplitude line). Useful for distinguishing between percussive and harmonic sounds.
    o	zero_crossing_rate
  •	Tempo: The beats-per-minute (BPM) or speed of the music.
    o	tempo
2. Spectral Features (features related to the frequency spectrum of the audio signal)
  •	Spectral Centroid: The "center of mass" of the spectrum, representing where the bulk of energy is concentrated (perceived brightness).
    o	spectral_centroid
  •	Spectral Bandwidth: Represents the width of the spectrum (i.e., the range of frequencies).
    o	spectral_bandwidth
  •	Spectral Contrast: Measures the difference in amplitude between peaks and valleys in a sound spectrum.
    o	spectral_contrast
  •	Spectral Rolloff: The frequency below which a certain percentage (typically 85%) of the total spectral energy lies. It gives an indication of the "sharpness" of the sound.
    o	spectral_rolloff
  •	Spectral Flatness: Describes how flat the spectrum is across the range of frequencies.
    o	spectral_flatness
3. Cepstral Features
  •	MFCCs (Mel-Frequency Cepstral Coefficients): Represents the short-term power spectrum of the audio, typically used in speech processing. Usually, 13–40 coefficients are extracted.
    o	mfcc_1, mfcc_2, ..., mfcc_13
4. Chroma Features (related to the 12 different pitch classes)
  •	Chroma Feature: Represents the 12 different pitch classes (C, C#, D, D#, etc.) in a 12-dimensional vector. Each dimension corresponds to one pitch class, capturing harmonic characteristics.
    o	chroma_c, chroma_c#, ..., chroma_b
  •	Chroma Energy Normalized (CENS): A smoothed version of chroma features, capturing tonal features over larger windows of time.
    o	cens_c, cens_c#, ..., cens_b
5. Rhythm Features
  •	Tempo: The estimated beats-per-minute (BPM) of the track.
    o	tempo
6. Mel-Spectrogram Features
  •	Mel-Spectrogram: A representation of the power spectrum of audio in the Mel scale, which mimics human auditory perception. Each time slice has a vector of Mel frequencies.
    o	mel_spectrogram
7. Formant Features (specific to speech processing)
  •	Formants (F1, F2, F3): Resonant frequencies of the vocal tract that distinguish different vowel sounds.
    o	formant_f1, formant_f2, formant_f3
8. Pitch Features
  •	Pitch: Fundamental frequency (F0) of the sound, representing the perceived pitch.
    o	pitch
  •	Pitch Confidence: The confidence level of the pitch estimation.
    o	pitch_confidence
9. Target 
   * Tells emotion of 8 {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }

Step-3 Model development and implementational steps :
1) Support vector machine (SVM) :
    1.Load the dataset into a Data Frame.
    2.Separate the features (X) and the target labels (y).
    3.Apply scaling to standardize the feature set.
    4.Define a dictionary to map target labels to emotion categories.
    5.Train the SVM model with K-Fold Cross Validation:
    6.Split the data into training and testing sets using K-Fold Cross Validation.
    7.Train an SVM model on the training data.
    8.Test the model on the testing data and calculate accuracy for each fold.
    9.Display a classification report and compute the average accuracy across all folds.
    10.Use the trained model to predict emotion from newly extracted features and map the
    11.predicted label to an emotion
   
2)Random forest classifier (RF) : 
    1.Create a list of relevant features for the model.
    2.Select only the available features in the dataset and define X (features) and y (target).
    3.Scale the feature matrix using a standard scaler.
    4.Select the top K features based on statistical significance.
    5.If any MFCC features are selected, include all MFCC features in the final feature set.
    6.Define the K-fold cross-validation strategy (with K=9).
    7.Train and evaluate the Random Forest model (without feature selection):
    8.Train the model on the training set and evaluate it on the testing set for each fold.
    9.Store accuracy and classification reports for each fold.
    10.Train and evaluate the Random Forest model (with feature selection):
    11.Train the model on the selected features and evaluate it across folds using crossvalidation.
    12.Store the classification reports and accuracies for each fold.
    13.Calculate and display the average classification report and accuracy across all folds

3)Long Short-Term Memory (LSTM) :
  1.Features (input) and labels (target) are separated from both training and testing data.
  2.LabelEncoder is used to encode the labels, and to_categorical converts them into a one-hotencoded format, necessary for categorical classification.
  3.Finally, you expand the dimensions of the feature arrays for compatibility with the LSTM,which expects 3D input (samples, timesteps, features).
  4.Model has layers : Standardizes the inputs, which can help stabilize the learning process.
  5.Three LSTM layers with 256 and 128 units, using L2 regularization to prevent overfitting. The return_sequences=True parameter ensures that each LSTM layer passes sequences of data to the next layer.
  6.Flattens the 3D LSTM output into 2D, so it can be passed into fully connected (Dense) layers.
  7.A fully connected layer with 8 units (or as per the number of classes) and a softmax activation for classification.
  8.The model is compiled using the categorical_crossentropy loss function, due to its ability to adapt learning rates during training

4) Multilayered Perceptron (MLP) :
  1.Convert the target labels into a one-hot encoded format to prepare them for classification.
  2.Split the dataset into training and testing sets, usually using an 80-20 split, with shuffling enabled for randomness.
  Scale the features using a standardization technique (e.g., StandardScaler) to normalize the data for better model performance.
  3.Create a function that defines the Multi-Layer Perceptron (MLP) model architecture. This includes:
      * Adding the input layer with a dynamic number of units.
      * Adding variable hidden layers, with each layer having a variable number of units.
  4.Defining the output layer with softmax activation for multi-class classification.
  5.Compiling the model with the Adam optimizer and categorical cross-entropy loss function.
  6.Use BayesianOptimization to search for the optimal hyperparameters (such as the number of layers, units, and learning rate) based on validation accuracy.
  7.Use the tuner to perform multiple trials to find the best combination of hyperparameters by evaluating on the training and validation sets.
  8.After the search, retrieve the optimal hyperparameters found during the tuning process.
  9.Use the optimal hyperparameters to build and train the final model on the training set.
  10.Evaluate the trained model's performance on the test set and print the test accuracy.
  11.Optionally, analyze the performance metrics (like accuracy) across different trials to assess improvements during hyperparameter tuning.

step-4 Results :  reports Screen shots 
  Support vector machine (SVM) :
  ![image](https://github.com/user-attachments/assets/54cf8fba-31fc-4b87-8946-a821c99bf271)

  Random forest Classifier (RF) :
  ![image](https://github.com/user-attachments/assets/be271480-1701-4014-9ca1-e6ade39cfe4f)

  Long Short-Term Memory (LSTM) :
  ![image](https://github.com/user-attachments/assets/aa265091-f24c-48e7-af66-a9d2aa609485)

  Multi layered perceptron (MLP):
  ![image](https://github.com/user-attachments/assets/cbdf28fa-08a6-4255-94a5-6ebd7ef4098b)


  


