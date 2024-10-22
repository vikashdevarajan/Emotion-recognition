import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('D:\\ML PROJECT\\dataset.csv')
correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()
# Zero Crossing Rate
    # MFCC
    # Chroma_stft
    # MelSpectogram
    # Spectral Centroid
    # Spectral Rolloff
    # Fourier Tempogram