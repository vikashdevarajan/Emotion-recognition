import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
def predict_emotion1(extracted_features):
    warnings.filterwarnings('ignore')
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    # Step 1: Load the data
    file_path = 'D:\\ML PROJECT\\dataset.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path)

    # Print the column names to check for missing features
    #print("Available columns:", data.columns.tolist())
    #print("First few rows of the dataset:")
    #print(data.head())

    # Step 3: Check for Missing Values
    #missing_values = data.isnull().sum()
    #print("\nMissing values in each column:")
    #print(missing_values[missing_values > 0])

    # Step 4: Handle Missing Values
    # Option 1: Drop rows with missing values
    #data_cleaned = data.dropna()

    # Option 2: Alternatively, you can impute missing values (uncomment if needed)
    # Example: Filling missing values with the mean of each column
    # data.fillna(data.mean(), inplace=True)

    # Step 5: Display the cleaned dataset
    #print("\nCleaned Dataset:")
    #print(data_cleaned.head())

    # Optional: Display summary information about the cleaned dataset
    #print("\nSummary of the cleaned dataset:")
    #print(data_cleaned.info())

    # Step 6: Exploratory Data Analysis (EDA)

    ## Summary Statistics
    #print("\nSummary Statistics:")
    #print(data_cleaned.describe())
    ### Distribution of Target Variable
    #plt.figure(figsize=(8, 5))
    #sns.countplot(x='target', data=data_cleaned)  # Adjust 'target' based on your actual target column name
    #plt.title('Distribution of Target Variable')
    #plt.xlabel('Target')
    #plt.ylabel('Count')
    #plt.show()

    ### Class Distribution Visualization (Bar Plot)
    '''plt.figure(figsize=(10, 6))
    sns.countplot(x='target', data=data_cleaned, palette='viridis')  # Count of each class in target variable
    plt.title('Class Distribution of Emotions')
    plt.xlabel('Emotion Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    correlation_matrix = data_cleaned.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Features')
    plt.show()

    ### Pairplot of Selected Features
    selected_features = ['zero_crossing_rate', 'rms_energy', 'spectral_centroid', 'target']  # Adjust as needed
    sns.pairplot(data_cleaned[selected_features], hue='target')
    plt.title('Pairplot of Selected Features')
    plt.show()'''
    feature_list = [
        'zero_crossing_rate', 'rms_energy', 'spectral_centroid', 'spectral_bandwidth',
         'spectral_rolloff', 'spectral_flatness', 'mfcc_1', 'mfcc_2',
        'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10',
        'mfcc_11', 'mfcc_12', 'mfcc_13',  'chroma_1', 'chroma_2', 
        'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 
        'chroma_10', 'chroma_11', 'chroma_12', 'cens_1', 'cens_2', 'cens_3', 'cens_4', 'cens_5', 
        'cens_6', 'cens_7', 'cens_8', 'cens_9', 'cens_10', 'cens_11', 'cens_12',  'tempo',  'pitch', 
        'pitch_confidence', 'mel_spectrogram', 'formant_1', 
        'formant_2', 'formant_3'
    ]

    # Filter out any features not present in the DataFrame
    features = [feature for feature in feature_list if feature in data.columns]
    print("Using features:", features,'\n')

    # Create the feature matrix and target variable from the DataFrame
    X = data[features].values
    y = data['target']  # Assuming the target variable is in this column

    # Step 3: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 4: Implement Feature Selection (SelectKBest)
    k_best_selector = SelectKBest(f_classif, k=10)  # Select top 10 features
    X_selected = k_best_selector.fit_transform(X_scaled, y)

    # Get the selected feature indices and their names
    selected_indices = k_best_selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_indices]

    # Check if any features were selected and print them
    if selected_features:
        print("Selected features:", selected_features,'\n\n')
    else:
        print("No features were selected.")

    # Step 5: Include all MFCC features if any MFCC feature is selected
    if any("mfcc" in feature for feature in selected_features):
        mfcc_features = [f'mfcc_{i}' for i in range(1, 14) if f'mfcc_{i}' in features]
        selected_features = list(set(selected_features) | set(mfcc_features))  # Add all mfcc features to selected features

    # Print the final selected features including all MFCC features if applicable
    print("Final Selected features (including all MFCCs if any were selected):", selected_features,'\n\n')
    
    # Create the final feature matrix based on selected features
    X_final_selected = data[selected_features].values

    # Step 6: Set up K-Fold Cross-Validation
    kf = KFold(n_splits=9, shuffle=True, random_state=42)  # K=9

    # Step 7: Train and evaluate the Random Forest model (without feature selection)
    from sklearn.metrics import classification_report, accuracy_score
    import numpy as np
    import pandas as pd

    # Step 7: Train and evaluate the Random Forest model (without feature selection)
    rf_reports = []  # Store classification reports for each fold
    rf_accuracies = []  # Store accuracy for each fold

    print("\nRandom Forest Model Evaluation (Without Feature Selection):")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize model
    for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), start=1):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf_model.fit(X_train, y_train)  # Train model
        
        # Predict and evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        rf_accuracies.append(accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        rf_reports.append(report)
        
        print(f"Fold {fold}: Accuracy: {accuracy:.2f}")

    # Calculate the average classification report across all folds
    average_report = {}

    # Get all classes from the first fold's classification report
    classes = list(rf_reports[0].keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    # Initialize average_report with zeros
    for label in classes:
        average_report[label] = {metric: 0.0 for metric in rf_reports[0][label]}

    # Sum the classification metrics over all folds
    for report in rf_reports:
        for label in classes:
            for metric in average_report[label]:
                average_report[label][metric] += report[label][metric]

    # Divide by the number of folds to get the average
    n_folds = len(rf_reports)
    for label in classes:
        for metric in average_report[label]:
            average_report[label][metric] /= n_folds

    # Print the average classification report
    print("\nAverage Classification Report (Across all folds):")
    for label, metrics in average_report.items():
        print(f"\nClass: {label}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

    # Calculate and print the average accuracy
    average_accuracy = np.mean(rf_accuracies)
    print(f"\nAverage Accuracy (Across all folds): {average_accuracy:.2f}")

    rf_fs_reports = []  # Store classification reports for each fold
    rf_fs_accuracies = []  # Store accuracy for each fold

    print("\nRandom Forest Model Evaluation (With Feature Selection):")
    for fold, (train_index, test_index) in enumerate(kf.split(X_final_selected), start=1):
        X_train, X_test = X_final_selected[train_index], X_final_selected[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rf_model.fit(X_train, y_train)  # Fit the model
        
        # Predict and evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        rf_fs_accuracies.append(accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        rf_fs_reports.append(report)
        
        print(f"Fold {fold}: Accuracy: {accuracy:.2f}")

    # Calculate the average classification report across all folds (with feature selection)
    average_rf_fs_report = {}

    # Get all classes from the first fold's classification report
    classes = list(rf_fs_reports[0].keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    # Initialize average_rf_fs_report with zeros
    for label in classes:
        average_rf_fs_report[label] = {metric: 0.0 for metric in rf_fs_reports[0][label]}

    # Sum the classification metrics over all folds
    for report in rf_fs_reports:
        for label in classes:
            for metric in average_rf_fs_report[label]:
                average_rf_fs_report[label][metric] += report[label][metric]

    # Divide by the number of folds to get the average
    n_folds = len(rf_fs_reports)
    for label in classes:
        for metric in average_rf_fs_report[label]:
            average_rf_fs_report[label][metric] /= n_folds

    # Print the average classification report
    print("\nAverage Classification Report (With Feature Selection, Across all folds):")
    for label, metrics in average_rf_fs_report.items():
        print(f"\nClass: {label}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

    # Calculate and print the average accuracy
    average_rf_fs_accuracy = np.mean(rf_fs_accuracies)
    print(f"\nAverage Random Forest Accuracy (With Feature Selection, Across all folds): {average_rf_fs_accuracy:.2f}")

    # Step 5: Use the trained model to predict the emotion from extracted features
    '''df_features_scaled = scaler.transform(df_features)
    emotion_prediction = rf_model.predict(df_features_scaled)
    predicted_emotion = emotions.get(str(emotion_prediction[0]).zfill(2), 'unknown')
    # Output the prediction
    print(f"The predicted emotion is: {predicted_emotion}")
    print(f'Predicted class: {emotion_prediction}')'''
        # Step 8: Predict the emotion using the trained Random Forest model
    df_features = pd.DataFrame([extracted_features])
    df_features_scaled = scaler.transform(df_features)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Predict using the trained model
    rf_model.fit(X_scaled, y)
    emotion_prediction = rf_model.predict(df_features_scaled)
    predicted_emotion = emotions.get(str(emotion_prediction[0]).zfill(2), 'unknown')
    return predicted_emotion




