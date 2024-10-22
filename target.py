import pandas as pd

# The emotion mapping dictionary
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

# Load the dataset.csv file
df = pd.read_csv('D:\\ML PROJECT\\dataset.csv')

# List to store the emotion codes (targets)
emotion_targets = []

# Process each filename to extract the emotion code and map it to the corresponding emotion
for filename in df['filename']:
    # Split the filename by '-' and get the second segment (emotion code)
    emotion_code = filename.split('-')[2]
    
    # Append the emotion code directly to the list (this is the "target")
    emotion_targets.append(emotion_code)

# Create a new DataFrame for the target emotion codes
target_df = pd.DataFrame({'filename': df['filename'], 'emotion_code': emotion_targets})

# Save the target information to target.csv
target_df.to_csv('D:\\ML PROJECT\\target.csv', index=False)

print("Target emotion codes have been successfully saved to 'target.csv'.")
