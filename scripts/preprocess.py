import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Euroleague data
file_path = 'data/euroleague_boxscores_all.csv'
data = pd.read_csv(file_path)

# 1. Data Cleaning: Remove unnecessary columns
data_cleaned = data.drop(columns=['Player_ID', 'Player', 'Dorsal', 'Minutes', 'Gamecode', 'Season'])

# Separate numeric and non-numeric columns
numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = data_cleaned.select_dtypes(exclude=['float64', 'int64']).columns

# 2. Handle Missing Values for numeric columns
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(data_cleaned[numeric_cols].mean())

# Non-numeric columns (like 'Team') can be handled separately
data_cleaned[non_numeric_cols] = data_cleaned[non_numeric_cols].fillna('Unknown')

# 3. Aggregate Data by Team
team_data = data_cleaned.groupby(['Team', 'IsPlaying']).agg({
    'Assistances': 'sum',
    'Steals': 'sum',
    'Turnovers': 'sum',
    'BlocksFavour': 'sum',
    'BlocksAgainst': 'sum',
    'FoulsCommited': 'sum',
    'FoulsReceived': 'sum',
    'Valuation': 'mean',  # Average valuation
    'Plusminus': 'sum',
    'Points': 'sum'  # Points for rolling window calculation
}).reset_index()

# 4. Create Rolling Features for Points
window_sizes = range(1, 11)

# Create rolling features for each window size (1-10)
for window_size in window_sizes:
    team_data[f'Rolling_Points_{window_size}'] = team_data['Points'].rolling(window=window_size).mean()

# Drop rows with NaN values resulting from the rolling operation
team_data.dropna(inplace=True)

# 5. Normalize the features
features = ['Assistances', 'Steals', 'Turnovers', 'BlocksFavour', 'BlocksAgainst', 
            'FoulsCommited', 'FoulsReceived', 'Valuation', 'Plusminus'] + [f'Rolling_Points_{window_size}' for window_size in window_sizes]

X = team_data[features].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and validation sets before creating sequences
train_size = int(0.8 * len(team_data))  # Use 80% for training, 20% for validation
X_train = X[:train_size]
X_valid = X[train_size:]
y_train = team_data['IsPlaying'].values[:train_size]
y_valid = team_data['IsPlaying'].values[train_size:]

# 6. Create sequences for each window size from 1 to 10
window_sizes = range(1, 11)

# Function to create sequences for a given window size
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Prepare sequences and save them for each window size
for window_size in window_sizes:
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
    X_valid_seq, y_valid_seq = create_sequences(X_valid, y_valid, window_size)

    # Save sequences for each window size
    np.save(f'data/X_train_window_{window_size}.npy', X_train_seq)
    np.save(f'data/y_train_window_{window_size}.npy', y_train_seq)
    np.save(f'data/X_valid_window_{window_size}.npy', X_valid_seq)
    np.save(f'data/y_valid_window_{window_size}.npy', y_valid_seq)

print("Data preprocessing for all window sizes (1-10) is complete!")
