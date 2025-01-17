import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiWindowClassifier:
    def __init__(self, window_sizes: List[int]):
        self.window_sizes = window_sizes
        self.scalers = {size: StandardScaler() for size in window_sizes}
        self.results = {
            'lstm': {},
            'gru': {}
        }

    def _build_model(self, model_type: str, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM or GRU model using functional API."""
        inputs = Input(shape=input_shape)
        layer_class = LSTM if model_type == 'lstm' else GRU

        x = layer_class(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = layer_class(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        return model

    def load_and_prepare_data(self, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare data for specific window size."""
        try:
            X_train = np.load(f'data/X_train_window_{window_size}.npy')
            y_train = np.load(f'data/y_train_window_{window_size}.npy')
            X_valid = np.load(f'data/X_valid_window_{window_size}.npy')
            y_valid = np.load(f'data/y_valid_window_{window_size}.npy')

            if X_train.size == 0 or X_valid.size == 0:
                raise ValueError(f"Empty data arrays for window size {window_size}")

            # Normalize features
            scaler = self.scalers[window_size]
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_valid_scaled = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)

            return X_train_scaled, y_train, X_valid_scaled, y_valid

        except Exception as e:
            logging.error(f"Error loading data for window size {window_size}: {e}")
            return None, None, None, None

    def train_and_evaluate(self, epochs: int = 15, batch_size: int = 32):
        """Train and evaluate models for all window sizes."""
        successful_windows = []

        for window_size in self.window_sizes:
            logging.info(f"\nProcessing window size: {window_size}")

            try:
                X_train, y_train, X_valid, y_valid = self.load_and_prepare_data(window_size)
                if X_train is None:
                    continue

                input_shape = (X_train.shape[1], X_train.shape[2])

                # Initialize results dictionary for this window size
                self.results['lstm'][window_size] = {}
                self.results['gru'][window_size] = {}

                # Train and evaluate LSTM
                lstm_model = self._build_model('lstm', input_shape)
                history_lstm = lstm_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_valid, y_valid),
                    verbose=0
                )

                # Train and evaluate GRU
                gru_model = self._build_model('gru', input_shape)
                history_gru = gru_model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_valid, y_valid),
                    verbose=0
                )

                # Store results - using MAE instead of accuracy
                self.results['lstm'][window_size] = {
                    'history': history_lstm.history,
                    'model': lstm_model,
                    'validation_mae': min(history_lstm.history['val_mae'])  # Using min since we want to minimize MAE
                }

                self.results['gru'][window_size] = {
                    'history': history_gru.history,
                    'model': gru_model,
                    'validation_mae': min(history_gru.history['val_mae'])
                }

                logging.info(f"Window {window_size} - LSTM Val MAE: {min(history_lstm.history['val_mae']):.4f}")
                logging.info(f"Window {window_size} - GRU Val MAE: {min(history_gru.history['val_mae']):.4f}")

                successful_windows.append(window_size)

            except Exception as e:
                logging.error(f"Error processing window size {window_size}: {e}")
                continue

        self.successful_windows = successful_windows

    def plot_results(self):
        """Plot comparison of results across successful window sizes."""
        if not hasattr(self, 'successful_windows') or not self.successful_windows:
            logging.error("No successful window sizes to plot")
            return

        plt.figure(figsize=(15, 5))

        # Plot 1: Best validation MAE per window size
        plt.subplot(1, 2, 1)
        lstm_mae = [self.results['lstm'][w]['validation_mae'] for w in self.successful_windows]
        gru_mae = [self.results['gru'][w]['validation_mae'] for w in self.successful_windows]

        plt.plot(self.successful_windows, lstm_mae, 'o-', label='LSTM')
        plt.plot(self.successful_windows, gru_mae, 'o-', label='GRU')
        plt.title('Best Validation MAE by Window Size')
        plt.xlabel('Window Size')
        plt.ylabel('Validation MAE')
        plt.legend()
        plt.grid(True)

        # Plot 2: Training curves for best window size
        plt.subplot(1, 2, 2)
        best_config = self.get_best_configuration()
        if best_config:
            best_window = best_config['window_size']
            best_model = best_config['model_type']

            plt.plot(self.results[best_model][best_window]['history']['mae'], 
                     label='Train')
            plt.plot(self.results[best_model][best_window]['history']['val_mae'], 
                     label='Validation')

            plt.title(f'Training History for Best Model\n({best_model.upper()}, Window={best_window})')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_best_configuration(self) -> Dict:
        """Find the best model configuration among successful runs."""
        if not hasattr(self, 'successful_windows') or not self.successful_windows:
            logging.error("No successful window sizes to evaluate")
            return None

        best_mae = float('inf')  # Initialize with infinity since we want to minimize MAE
        best_config = {}

        for model_type in ['lstm', 'gru']:
            for window_size in self.successful_windows:
                mae = self.results[model_type][window_size]['validation_mae']
                if mae < best_mae:  # Changed from > to < since we want to minimize MAE
                    best_mae = mae
                    best_config = {
                        'model_type': model_type,
                        'window_size': window_size,
                        'mae': mae,
                        'model': self.results[model_type][window_size]['model']
                    }

        return best_config

    def save_best_model(self):
        """Save the best performing model."""
        best_config = self.get_best_configuration()
        if best_config:
            model = best_config['model']
            model_type = best_config['model_type']
            window_size = best_config['window_size']
            mae = best_config['mae']

            # Create model name with details
            model_name = f"best_model_{model_type}_window{window_size}_mae{mae:.4f}"

            try:
                # Save full model in .keras format
                model.save(f'models/{model_name}.keras')

                # Save model architecture as JSON
                model_json = model.to_json()
                with open(f'models/{model_name}_architecture.json', 'w') as f:
                    f.write(model_json)

                # Save model weights in separate .weights format
                model.save_weights(f'models/{model_name}_weights.weights.h5')

                logging.info(f"\nModel saved successfully as '{model_name}'")
                logging.info(f"Files saved:")
                logging.info(f"- Full model: {model_name}.keras")
                logging.info(f"- Architecture: {model_name}_architecture.json")
                logging.info(f"- Weights: {model_name}_weights.weights.h5")

            except Exception as e:
                logging.error(f"Error saving model: {e}")
                return None

            return model_name
        return None

def main():
    # Define window sizes to evaluate
    window_sizes = list(range(1, 9))  # Evaluate all window sizes from 1 to 10

    # Create models directory if it doesn't exist
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    # Initialize and run classifier
    classifier = MultiWindowClassifier(window_sizes)
    classifier.train_and_evaluate(epochs=15, batch_size=32)

    # Plot and display results
    classifier.plot_results()

    # Print best configuration and save model
    best_config = classifier.get_best_configuration()
    if best_config:
        logging.info("\nBest Configuration:")
        logging.info(f"Model Type: {best_config['model_type'].upper()}")
        logging.info(f"Window Size: {best_config['window_size']}")
        logging.info(f"Validation MAE: {best_config['mae']:.4f}")

        # Save the best model
        saved_model_name = classifier.save_best_model()

if __name__ == "__main__":
    main()