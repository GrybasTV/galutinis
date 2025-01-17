import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the best model
best_model = tf.keras.models.load_model('models/best_model_gru_window8_mae0.0161.keras')

# Load the Euroleague data
file_path = 'data/euroleague_boxscores_all.csv'
data = pd.read_csv(file_path)

# Data Preprocessing
data_cleaned = data.drop(columns=['Player_ID', 'Player', 'Dorsal', 'Minutes', 'Gamecode', 'Season'])
numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = data_cleaned.select_dtypes(exclude=['float64', 'int64']).columns

# Fill missing values
data_cleaned[numeric_cols] = data_cleaned[numeric_cols].fillna(0)
data_cleaned[non_numeric_cols] = data_cleaned[non_numeric_cols].fillna('Unknown')

# First group by Team and Game to get max points per game
game_points = data_cleaned.groupby(['Team', 'IsPlaying'])['Points'].max().reset_index()

# Then group by Team and Game to get other game-level stats (using mean for team stats)
game_stats_temp = data_cleaned.groupby(['Team', 'IsPlaying']).agg({
    'Assistances': 'mean',
    'Steals': 'mean',
    'Turnovers': 'mean',
    'BlocksFavour': 'mean',
    'BlocksAgainst': 'mean',
    'FoulsCommited': 'mean',
    'FoulsReceived': 'mean',
    'Valuation': 'mean',
    'Plusminus': 'mean',
    'OffensiveRebounds': 'mean',
    'DefensiveRebounds': 'mean',
    'TotalRebounds': 'mean',
    'FieldGoalsMade2': 'mean',
    'FieldGoalsAttempted2': 'mean',
    'FieldGoalsMade3': 'mean',
    'FieldGoalsAttempted3': 'mean',
    'FreeThrowsMade': 'mean',
    'FreeThrowsAttempted': 'mean'
}).reset_index()

# Merge the points with other stats
game_stats = game_points.merge(game_stats_temp, on=['Team', 'IsPlaying'])

# Calculate additional metrics
game_stats['FG2_Percentage'] = (game_stats['FieldGoalsMade2'] / game_stats['FieldGoalsAttempted2'] * 100).fillna(0)
game_stats['FG3_Percentage'] = (game_stats['FieldGoalsMade3'] / game_stats['FieldGoalsAttempted3'] * 100).fillna(0)
game_stats['FT_Percentage'] = (game_stats['FreeThrowsMade'] / game_stats['FreeThrowsAttempted'] * 100).fillna(0)

# Calculate season averages for each team
season_averages = game_stats.groupby('Team').agg({
    'Points': 'mean',
    'FG2_Percentage': 'mean',
    'FG3_Percentage': 'mean',
    'FT_Percentage': 'mean',
    'Assistances': 'mean',
    'TotalRebounds': 'mean',
    'Steals': 'mean',
    'BlocksFavour': 'mean',
    'Turnovers': 'mean'
}).round(1)

# Define features in the correct order
features = [
    'Points', 'Assistances', 'Steals', 'Turnovers',
    'BlocksFavour', 'BlocksAgainst', 'FoulsCommited', 'FoulsReceived',
    'Valuation', 'Plusminus', 'OffensiveRebounds', 'DefensiveRebounds',
    'TotalRebounds', 'FG2_Percentage', 'FieldGoalsAttempted2',
    'FG3_Percentage', 'FieldGoalsAttempted3', 'FT_Percentage',
    'FreeThrowsAttempted'
]

# Initialize scaler
scaler = StandardScaler()
all_feature_data = game_stats[features].values
scaler.fit(all_feature_data)

def prepare_team_sequence(team_df, window_size=8):
    """
    Prepare sequence for a team with proper error handling and padding if necessary
    """
    # Get recent games data
    recent_data = team_df[features].tail(window_size)
    
    # Check if we have enough data
    if len(recent_data) < window_size:
        # If not enough data, pad with means
        padding_needed = window_size - len(recent_data)
        mean_values = recent_data.mean()
        padding = pd.DataFrame([mean_values] * padding_needed)
        recent_data = pd.concat([padding, recent_data], ignore_index=True)
    
    # Convert to numpy array and scale
    sequence = recent_data.values
    scaled_sequence = scaler.transform(sequence)
    
    # Reshape to model's expected input shape
    return scaled_sequence.reshape(1, window_size, len(features))

# Streamlit UI
st.title('Euroleague Game Prediction')
teams = sorted(game_stats['Team'].unique())

# Team selection
team_a = st.selectbox('Select first team:', teams)
team_b = st.selectbox('Select second team:', teams)

if team_a != team_b:
    try:
        # Filter data for selected teams
        team_a_data = game_stats[game_stats['Team'] == team_a].copy()
        team_b_data = game_stats[game_stats['Team'] == team_b].copy()
        
        if len(team_a_data) == 0 or len(team_b_data) == 0:
            st.error("No data available for one or both teams.")
        else:
            # Prepare sequences
            team_a_seq = prepare_team_sequence(team_a_data)
            team_b_seq = prepare_team_sequence(team_b_data)
            
            # Make predictions
            pred_a = best_model.predict(team_a_seq, verbose=0)
            pred_b = best_model.predict(team_b_seq, verbose=0)
            
            # Extract predictions and scale to reasonable point ranges
            base_points = 75  # Average Euroleague game points
            point_scaling = 20  # Scaling factor for variations
            
            predicted_points_a = base_points + (float(pred_a[0]) * point_scaling)
            predicted_points_b = base_points + (float(pred_b[0]) * point_scaling)
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(team_a)
                st.metric("Predicted Points", f"{int(round(predicted_points_a))}")
                win_prob_a = 1 / (1 + np.exp((predicted_points_b - predicted_points_a) / 10))
                st.metric("Win Probability", f"{win_prob_a:.1%}")
                
                # Show recent form
                st.write("Recent Points:")
                recent_points_a = team_a_data['Points'].tail(5).values[::-1]  # Reverse order
                for i, points in enumerate(recent_points_a, 1):
                    st.write(f"Game {i}: {points:.0f}")
            
            with col2:
                st.subheader(team_b)
                st.metric("Predicted Points", f"{int(round(predicted_points_b))}")
                win_prob_b = 1 - win_prob_a
                st.metric("Win Probability", f"{win_prob_b:.1%}")
                
                # Show recent form
                st.write("Recent Points:")
                recent_points_b = team_b_data['Points'].tail(5).values[::-1]  # Reverse order
                for i, points in enumerate(recent_points_b, 1):
                    st.write(f"Game {i}: {points:.0f}")
            
            # Add season averages comparison
            st.subheader("Team Season Averages Comparison")
            display_features = [
                'Points', 'FG2_Percentage', 'FG3_Percentage', 'FT_Percentage',
                'Assistances', 'TotalRebounds', 'Steals', 'BlocksFavour',
                'Turnovers'
            ]
            
            stats_comparison = pd.DataFrame({
                team_a: season_averages.loc[team_a, display_features],
                team_b: season_averages.loc[team_b, display_features]
            })
            
            # Rename index for better display
            feature_names = {
                'Points': 'Points Per Game',
                'FG2_Percentage': '2PT %',
                'FG3_Percentage': '3PT %',
                'FT_Percentage': 'FT %',
                'Assistances': 'Assists',
                'TotalRebounds': 'Rebounds',
                'Steals': 'Steals',
                'BlocksFavour': 'Blocks',
                'Turnovers': 'Turnovers'
            }
            stats_comparison.index = [feature_names[feat] for feat in display_features]
            
            st.dataframe(stats_comparison.style.format("{:.1f}"))
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try different teams or check if enough historical data is available.")
else:
    st.warning("Please select different teams!")

# Add footer with model information
st.markdown("---")
st.caption("Model: GRU Neural Network | Window Size: 8 games | MAE: 0.0161")
