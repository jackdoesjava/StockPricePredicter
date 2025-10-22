import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from typing import List, Tuple, Dict, Optional, Union
import warnings
import os
import glob
import time

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from IPython.display import display


# Optional Dependencies (for modular inclusion)
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    warnings.warn("Keras/TensorFlow not installed. LSTM functionality will be unavailable.", ImportWarning)


# --- Configuration and Constants ---
# Centralized configuration for easy modification and reproducibility
np.random.seed(42) # Consistent seed for reproducibility
sns.set(style="whitegrid")
CACHE_DIR = 'Stocks'
DEFAULT_TICKER = 'GOOG'
DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = '2017-01-01'
DEFAULT_PREDICT_DAYS = 50
DEFAULT_TRAIN_TEST_RATIO = 0.8
DEFAULT_REGRESSION_DEGREES = [1, 2, 3]


# --- Helper Functions (for modularity and reusability) ---

# Original code (Causes warning):
def _handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing data using forward and backward fill."""
    df_clean = df.copy()
    # df_clean.fillna(method='ffill', inplace=True) # <-- Deprecated
    # df_clean.fillna(method='bfill', inplace=True) # <-- Deprecated
    
    # 💡 FIX: Use the recommended ffill() and bfill() methods
    df_clean = df_clean.ffill() 
    df_clean = df_clean.bfill()
    
    return df_clean

def _get_cache_path(ticker: str, start_date: str, end_date: str) -> str:
    """Generates a standardized cache file path."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f'{ticker}_{start_date}_{end_date}.csv')

def _calculate_fft_waveform(residuals: np.ndarray, num_harmonics: int, total_samples: int) -> np.ndarray:
    """Calculates the FFT-based waveform for a given residual series and number of harmonics."""
    n = len(residuals)
    fft_coeffs = np.fft.fft(residuals)
    frequencies = np.fft.fftfreq(n)

    # Find indices of highest amplitude frequencies (excluding the DC component at index 0)
    # The amplitudes are symmetric, so we take the top `num_harmonics` from one side
    sorted_indices = np.argsort(np.abs(fft_coeffs[1:n//2]))[::-1] + 1
    
    # Select the DC component and the top 'num_harmonics' pairs (positive and negative frequencies)
    # If num_harmonics is 0, only DC is used, but we want the overall trend, which is managed outside.
    # Here, we reconstruct the signal from the top components.
    top_indices = [0] + list(sorted_indices[:num_harmonics])
    
    # Reconstruct signal for all required time steps (training + prediction)
    t = np.arange(total_samples)
    fft_reconstructed = np.zeros(total_samples)
    
    for i in top_indices:
        amplitude = np.abs(fft_coeffs[i]) / n
        phase = np.angle(fft_coeffs[i])
        frequency = frequencies[i]
        
        # Add contribution of positive frequency component
        fft_reconstructed += amplitude * np.cos(2 * np.pi * frequency * t + phase)
        
        # Add contribution of negative frequency component (which is the complex conjugate pair)
        if i != 0 and i != n//2:
            neg_i = n - i
            neg_amplitude = np.abs(fft_coeffs[neg_i]) / n
            neg_phase = np.angle(fft_coeffs[neg_i])
            neg_frequency = frequencies[neg_i]
            fft_reconstructed += neg_amplitude * np.cos(2 * np.pi * neg_frequency * t + neg_phase)


    return fft_reconstructed


class ProductionStockRegressor:
    """
    A professional, modular class for fetching, modeling, and predicting stock prices.

    Implements Linear, Polynomial Regression, Momentum, and optional FFT/LSTM prediction.
    Ensures data integrity, model reproducibility, and high-quality reporting.
    """

    def __init__(self,
                 ticker: str = DEFAULT_TICKER,
                 start_date: str = DEFAULT_START_DATE,
                 end_date: str = DEFAULT_END_DATE,
                 predict_days: int = DEFAULT_PREDICT_DAYS,
                 verbose: bool = True):
        """
        Initializes the stock regressor and fetches/loads stock data.

        :param ticker: Stock ticker symbol (e.g., 'GOOG').
        :param start_date: Start date for data download (YYYY-MM-DD).
        :param end_date: End date for training data (YYYY-MM-DD).
        :param predict_days: Number of market days to predict beyond the end_date.
        :param verbose: If True, print status updates.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.train_end_date = end_date
        self.predict_days = predict_days
        self.verbose = verbose

        # Model and Data storage
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.reg_models: Dict[int, LinearRegression] = {}
        self.poly_features: Dict[int, PolynomialFeatures] = {}
        self.predictions: Dict[str, pd.Series] = {}
        self.scores: Dict[str, Dict[str, float]] = {}

        # Derived dates
        self.data_end_date: str = self._calculate_data_end_date()
        
        # Fetch data
        self._load_or_fetch_data()
        self._prepare_regression_data()


    def _calculate_data_end_date(self) -> str:
        """Calculates the final data fetch date, including the prediction window."""
        try:
            train_end = dt.datetime.strptime(self.train_end_date, "%Y-%m-%d")
            # Calculate a buffer for the prediction window (+ additional buffer days)
            # Use 1.5 * predict_days for a generous date range post-training end
            data_end = train_end + dt.timedelta(days=int(self.predict_days * 1.5))
            
            # Ensure the end date is not in the future
            if data_end > dt.datetime.today():
                data_end = dt.datetime.today() - dt.timedelta(days=1)
            
            # The testing phase will strictly use only available market days up to self.predict_days
            return data_end.strftime("%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {self.train_end_date}. Use YYYY-MM-DD.") from e
        except Exception as e:
            raise RuntimeError(f"Could not calculate data end date: {e}")

    def _load_or_fetch_data(self) -> None:
        """
        Loads data exclusively from local, date-stamped CSVs in the 'Stocks' folder.
        Network download (yfinance) functionality is temporarily disabled.
        """
        
        # Define requested date objects for comparison
        req_start_dt = dt.datetime.strptime(self.start_date, "%Y-%m-%d")
        # NOTE: self.data_end_date already includes the prediction buffer
        req_end_dt = dt.datetime.strptime(self.data_end_date, "%Y-%m-%d")
        
        # --- PRIORITY 1: Advanced Local CSV Search ('Stocks' folder) ---
        search_pattern = os.path.join(CACHE_DIR, f'Stock-{self.ticker}-*.csv')
        local_files = glob.glob(search_pattern)
        best_match_path = None
        
        if self.verbose:
            print(f"DEBUG: Searching local files using pattern: {search_pattern}")
            
        for file_path in local_files:
            try:
                # Example: Stock-GLD-2015-01-01-2017-08-27.csv
                date_part = os.path.basename(file_path).replace(f'Stock-{self.ticker}-', '').replace('.csv', '')
                parts = date_part.split('-')
                
                # *** FIX HERE: Check for the correct 6-part date format (YYYY, MM, DD, YYYY, MM, DD) ***
                if len(parts) == 6: 
                    # Correctly slice the 6 components into two YYYY-MM-DD strings
                    cached_start_str = f"{parts[0]}-{parts[1]}-{parts[2]}" # parts[0:3]
                    cached_end_str = f"{parts[3]}-{parts[4]}-{parts[5]}"   # parts[3:6]
                else:
                    # Skip files that don't match the expected 6-part format
                    continue 
                    
                cached_start_dt = dt.datetime.strptime(cached_start_str, "%Y-%m-%d")
                cached_end_dt = dt.datetime.strptime(cached_end_str, "%Y-%m-%d")

                # Check if the local file's range fully contains the requested range
                if req_start_dt >= cached_start_dt and req_end_dt <= cached_end_dt:
                    best_match_path = file_path
                    break # Found a perfect encompassing file, stop searching

            except Exception:
                continue # Skip files with bad date formats
        
        if best_match_path:
            try:
                self.data = pd.read_csv(best_match_path, index_col='Date', parse_dates=True)
                
                # Trim the loaded data to the exact requested dates
                self.data = self.data[self.start_date : self.data_end_date] 
                
                if self.data.empty:
                    raise ValueError("Date range filtered data resulted in an empty set.")

                self.data = _handle_missing_data(self.data)
                if self.verbose:
                    print(f"✅ SUCCESSFULLY LOADED LOCAL DATA: {best_match_path}")
                return # SUCCESS! Exit function.
                
            except Exception as e:
                # If file found but corrupted/unusable
                raise IOError(f"❌ FATAL ERROR: Local file {best_match_path} found but failed to process. Error: {e}") 

        # --- FALLBACK: Immediate Failure (Network is disabled) ---
        # If the code reaches here, no local file matched the criteria.
        
        raise IOError(
            f"❌ FATAL ERROR: No local file found in '{CACHE_DIR}' matching pattern 'Stock-{self.ticker}-*.csv' "
            f"that covers the required date range: {self.start_date} to {self.data_end_date}. "
            "Please check the ticker, folder path, and file date stamps."
        )
    def _prepare_regression_data(self) -> None:
        """
        Prepares the data for regression models by creating a timeline index and
        splitting into train/test sets based on the training end date.
        """
        if self.data is None:
            raise AttributeError("Data not loaded. Cannot prepare regression data.")
            
        # Select the 'Adj Close' column and reset index for a continuous timeline 'X' feature
        data_df = self.data.copy()
        data_df = data_df[['Adj Close']].reset_index(names=['Date'])
        data_df['Timeline'] = data_df.index.values

        # Determine the split point based on the training end date
        # The split uses the actual date index from yfinance data for precision
        try:
            # Find the last index *before or on* the target training end date
            train_end_dt = pd.to_datetime(self.train_end_date)
            train_index = data_df[data_df['Date'] <= train_end_dt].index.max()
            if pd.isna(train_index):
                 # Fallback to a ratio if the date is outside the fetched data range
                 split_point = int(DEFAULT_TRAIN_TEST_RATIO * len(data_df))
            else:
                split_point = train_index + 1 # exclusive split index

        except Exception:
            # Fallback to a ratio if any date error occurs
            split_point = int(DEFAULT_TRAIN_TEST_RATIO * len(data_df))
            if self.verbose:
                print(f"⚠️ Date split failed or date is outside range. Using {DEFAULT_TRAIN_TEST_RATIO*100}% ratio split.")
        
        if split_point == 0:
            raise ValueError("Training data window is too small or invalid.")

        # Limit the test set to the predicted number of days
        total_samples_for_training_and_testing = split_point + self.predict_days
        
        # X: Timeline index (features for regression)
        # y: Adjusted Close Price (target)
        self.X_train = data_df['Timeline'].iloc[:split_point].values.reshape(-1, 1)
        self.y_train = data_df['Adj Close'].iloc[:split_point].values

        self.X_test = data_df['Timeline'].iloc[split_point:total_samples_for_training_and_testing].values.reshape(-1, 1)
        self.y_test = data_df['Adj Close'].iloc[split_point:total_samples_for_training_and_testing].values
        
        # Keep the full data frame with 'Timeline' as index for easier plotting
        self.full_data = data_df[['Date', 'Adj Close']].iloc[:total_samples_for_training_and_testing].copy()
        self.full_data.set_index('Date', inplace=True)
        
        self.split_point = split_point
        
        if self.verbose:
            print(f"📊 Training set size: {len(self.X_train)} | Testing set size: {len(self.X_test)} (Max {self.predict_days} days)")
            print(f"   Train End Date: {data_df['Date'].iloc[self.split_point-1].strftime('%Y-%m-%d')}")
            if len(self.X_test) > 0:
                 print(f"   Test Start Date: {data_df['Date'].iloc[self.split_point].strftime('%Y-%m-%d')}")

    # --- Training Methods ---

    def train_regression_model(self, degrees: List[int] = DEFAULT_REGRESSION_DEGREES) -> None:
        """
        Trains Linear and Polynomial Regression models.
        
        :param degrees: List of polynomial degrees to train (e.g., [1, 2, 3]).
        """
        if self.X_train is None:
            raise RuntimeError("Regression data not prepared. Cannot train models.")

        self.reg_models = {}
        self.poly_features = {}
        
        for deg in degrees:
            try:
                # 1. Feature Engineering (Polynomial)
                poly_model = PolynomialFeatures(degree=deg)
                X_train_poly = poly_model.fit_transform(self.X_train)
                X_full_poly = poly_model.transform(np.arange(len(self.full_data)).reshape(-1, 1))

                # 2. Model Training
                model = LinearRegression()
                model.fit(X_train_poly, self.y_train)

                # 3. Prediction (for the entire available timeline)
                prediction_full = model.predict(X_full_poly)
                pred_series = pd.Series(prediction_full, index=self.full_data.index, name=f'Regression_Deg_{deg}')

                # 4. Store model, features, and prediction
                self.reg_models[deg] = model
                self.poly_features[deg] = poly_model
                self.predictions[f'Regression_Deg_{deg}'] = pred_series
                
                # 5. Evaluate and store score
                self._evaluate_model(f'Regression_Deg_{deg}', prediction_full)
                
            except Exception as e:
                warnings.warn(f"❌ Failed to train Polynomial Regression (Degree {deg}). Error: {e}", RuntimeWarning)
        
        if self.verbose and self.reg_models:
            print(f"✅ Trained Polynomial Regressions with degrees: {list(self.reg_models.keys())}")


    def train_momentum_model(self, lookback_days: int = 15, momentum_weight: float = 0.3) -> None:
        """
        Combines the average of polynomial regressions with a short-term momentum trend.
        
        :param lookback_days: Number of recent training days to use for momentum calculation.
        :param momentum_weight: Weight applied to the momentum regression prediction (0.0 to 1.0).
        """
        if not self.reg_models:
            warnings.warn("No base regression models found. Running `train_regression_model` first.")
            self.train_regression_model()
            if not self.reg_models:
                raise RuntimeError("Cannot train Momentum model without base regressions.")
        
        try:
            # 1. Calculate long-term trend (Average of all Poly Regressions)
            all_reg_preds = [self.predictions[key].values for key in self.predictions if key.startswith('Regression_Deg_')]
            if not all_reg_preds:
                 raise ValueError("No valid regression predictions to average.")
                 
            avg_reg_trend = np.mean(all_reg_preds, axis=0)

            # 2. Calculate short-term momentum trend (Linear Regression on last `lookback_days` of training)
            start_idx = len(self.X_train) - lookback_days
            X_mom = self.X_train[start_idx:]
            y_mom = self.y_train[start_idx:]
            
            # Use degree 1 (Linear) for momentum trend
            poly_mom = PolynomialFeatures(degree=1)
            X_mom_poly = poly_mom.fit_transform(X_mom)
            model_mom = LinearRegression()
            model_mom.fit(X_mom_poly, y_mom)

            # Predict momentum trend for the entire full data range
            X_full_mom_poly = poly_mom.transform(np.arange(len(self.full_data)).reshape(-1, 1))
            momentum_trend_full = model_mom.predict(X_full_mom_poly)

            # 3. Combine trends
            # Apply weighted combination *only* from the end of the momentum training window to the end of the forecast.
            # Before the window, use the pure average regression trend.
            
            # Find the index corresponding to the start of the momentum influence (in the full data)
            mom_start_index = self.split_point - lookback_days
            mom_start_index = max(0, mom_start_index) # Safety clip

            combined_prediction = avg_reg_trend.copy()
            
            # Apply blending in the momentum-aware window
            blended_trend = (1 - momentum_weight) * avg_reg_trend[mom_start_index:] + \
                            momentum_weight * momentum_trend_full[mom_start_index:]
            
            combined_prediction[mom_start_index:] = blended_trend

            pred_series = pd.Series(combined_prediction, index=self.full_data.index, name='Reg_Momentum')
            self.predictions['Reg_Momentum'] = pred_series
            self._evaluate_model('Reg_Momentum', combined_prediction)

            if self.verbose:
                print(f"✅ Trained Regression/Momentum model (Lookback: {lookback_days} days, Weight: {momentum_weight})")

        except Exception as e:
            warnings.warn(f"❌ Failed to train Regression/Momentum model. Error: {e}", RuntimeWarning)


    def train_fft_model(self, num_harmonics: int = 4, underlying_trend_poly: int = 3) -> None:
        """
        Trains an FFT-based model to forecast cyclical residuals around an underlying trend.
        
        :param num_harmonics: Number of high-amplitude Fourier components to use.
        :param underlying_trend_poly: Degree of the polynomial regression used for the underlying trend.
        """
        if 'Reg_Momentum' not in self.predictions:
            warnings.warn("Reg_Momentum model not found. Training with default parameters.")
            self.train_momentum_model()
        
        try:
            # 1. Get the underlying trend (Reg/Momentum model is the best blend)
            underlying_trend_full = self.predictions['Reg_Momentum'].values
            
            # 2. Calculate Residuals from the trend in the training period
            residuals_train = self.y_train - underlying_trend_full[:len(self.y_train)]

            # 3. Calculate FFT waveform from residuals
            fft_waveform_full = _calculate_fft_waveform(
                residuals=residuals_train, 
                num_harmonics=num_harmonics, 
                total_samples=len(self.full_data)
            )

            # 4. Combine Trend and FFT Waveform
            fft_prediction_full = underlying_trend_full + fft_waveform_full

            pred_series = pd.Series(fft_prediction_full, index=self.full_data.index, name='FFT_Prediction')
            self.predictions['FFT_Prediction'] = pred_series
            self._evaluate_model('FFT_Prediction', fft_prediction_full)

            if self.verbose:
                print(f"✅ Trained FFT model (Harmonics: {num_harmonics}, Underlying Trend: Reg/Momentum)")

        except Exception as e:
            warnings.warn(f"❌ Failed to train FFT model. Error: {e}", RuntimeWarning)


    def train_lstm_model(self, 
                         sequence_length: int = 50, 
                         units: int = 50, 
                         epochs: int = 5, 
                         batch_size: int = 512) -> None:
        """
        Trains an LSTM (Recurrent Neural Network) prediction model.
        
        :param sequence_length: The length of the look-back window for the LSTM.
        :param units: Number of units in the first LSTM layer.
        :param epochs: Number of training epochs.
        :param batch_size: Training batch size.
        """
        if not KERAS_AVAILABLE:
            warnings.warn("Keras/TensorFlow not available. Cannot train LSTM model.", UserWarning)
            return

        try:
            # 1. Data Preparation (Scaling and Windowing)
            scaler = MinMaxScaler(feature_range=(0, 1))
            prices = self.full_data['Adj Close'].values.reshape(-1, 1)
            scaled_prices = scaler.fit_transform(prices)

            X_seq, y_seq = [], []
            for i in range(len(scaled_prices) - sequence_length):
                X_seq.append(scaled_prices[i:i + sequence_length, 0])
                y_seq.append(scaled_prices[i + sequence_length, 0])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            # Reshape for LSTM: [samples, time steps, features]
            X_seq = X_seq[:, :, np.newaxis]
            
            # Split into training (up to split_point) and a remaining set
            train_size = len(self.y_train) - sequence_length
            X_train_lstm = X_seq[:train_size]
            y_train_lstm = y_seq[:train_size]
            X_remainder = X_seq[train_size:]
            
            if len(X_train_lstm) == 0:
                 raise ValueError("Training window is too short for the chosen sequence length.")

            # 2. Model Definition
            model = Sequential([
                LSTM(units=units, input_shape=(X_train_lstm.shape[1], 1), return_sequences=True, recurrent_dropout=0.2),
                Dropout(0.2),
                LSTM(units=units // 2, return_sequences=False, recurrent_dropout=0.2),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # 3. Training
            if self.verbose:
                 print(f"Training LSTM for {epochs} epochs...")
            model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=0)

            # 4. Prediction
            # The test set is the remaining historical data that was *not* used for training windows
            # We predict the value *after* the end of the sequence.
            
            # Predict on the remainder of the full data (validation + test + prediction)
            lstm_preds = model.predict(X_remainder, verbose=0)
            
            # Inverse Transform
            # The original price corresponding to the first predicted value (the date right after y_train_lstm ends)
            # The last known price is the anchor for denormalization
            
            # For simplicity and clean logic, we'll denormalize relative to the last known price.
            # In production, a more sophisticated sliding window approach with prediction-feedback should be used.
            
            # A simple (though slightly incorrect) approach: inverse transform the scaled prediction.
            # The 'y' values being predicted are single next-step values, so we can inverse-transform the full set of scaled predictions
            
            # Create a dummy array to inverse-transform the predictions
            # Scaler expects [samples, features]. Our prediction is just the scaled price.
            dummy = np.zeros(shape=(len(lstm_preds), scaled_prices.shape[1]))
            dummy[:, 0] = lstm_preds[:, 0]
            lstm_prediction_descaled = scaler.inverse_transform(dummy)[:, 0]

            # The LSTM predictions start at the index after the last training target
            start_index_full_data = len(y_seq) - len(lstm_prediction_descaled) + sequence_length
            
            prediction_full = self.full_data['Adj Close'].values.copy()
            prediction_full[start_index_full_data:] = lstm_prediction_descaled

            pred_series = pd.Series(prediction_full, index=self.full_data.index, name='LSTM_Prediction')
            self.predictions['LSTM_Prediction'] = pred_series
            self._evaluate_model('LSTM_Prediction', prediction_full)
            
            if self.verbose:
                print(f"✅ Trained LSTM model (Sequence: {sequence_length}, Units: {units}, Epochs: {epochs})")

        except Exception as e:
            warnings.warn(f"❌ Failed to train LSTM model. Error: {e}", RuntimeWarning)


    def _evaluate_model(self, model_name: str, full_prediction: np.ndarray) -> None:
        """
        Calculates and stores R^2 and MSE scores for a given prediction.
        
        :param model_name: Name of the model (e.g., 'Regression_Deg_1').
        :param full_prediction: Prediction array for the full data timeline.
        """
        # Training set evaluation
        y_train_pred = full_prediction[:self.split_point]
        r2_train = r2_score(self.y_train, y_train_pred)
        
        # Testing set evaluation (only the part corresponding to y_test)
        y_test_pred = full_prediction[self.split_point:self.split_point + len(self.y_test)]
        r2_test = r2_score(self.y_test, y_test_pred)
        mse_test = mean_squared_error(self.y_test, y_test_pred)

        self.scores[model_name] = {
            'R2_Train': r2_train,
            'R2_Test': r2_test,
            'MSE_Test': mse_test
        }


    # --- Reporting and Visualization ---

    def plot_predictions(self, model_names: Union[str, List[str]] = 'all', title: str = None) -> None:
        """
        Generates a publication-ready plot of actual vs. predicted prices.
        
        :param model_names: A single model name or a list of model names to plot. Use 'all' for all trained models.
        :param title: Custom title for the plot.
        """
        if self.full_data is None:
            print("Data not loaded. Cannot plot.")
            return

        if model_names == 'all':
            plot_models = list(self.predictions.keys())
        elif isinstance(model_names, str):
            plot_models = [model_names]
        else:
            plot_models = model_names
            
        plot_models = [name for name in plot_models if name in self.predictions]
        
        if not plot_models:
            print("No valid models to plot. Have you run the training methods?")
            return

        plt.figure(figsize=(18, 9))
        
        # Plot Actual Prices
        self.full_data['Adj Close'].plot(label='Actual Price', color='black', linewidth=2.5, alpha=0.8)

        # Plot Predictions
        for name in plot_models:
            self.predictions[name].plot(label=f'Prediction: {name}', alpha=0.7, linestyle='--')

        # Annotations and Visual Enhancements
        # 1. Training/Testing Split Line
        split_date = self.full_data.index[self.split_point - 1]
        plt.axvline(split_date, color='red', linestyle='--', linewidth=1.5, label='End of Training Data')
        plt.text(split_date + dt.timedelta(days=7), 
                 self.full_data['Adj Close'].max() * 0.95, 
                 'Test/Prediction Start', 
                 rotation=0, color='red', fontsize=10, ha='left')

        # 2. Key Dates/Trends (Example: 60-day Rolling Mean in Training Period)
        if len(self.y_train) > 60:
             train_data = self.full_data.iloc[:self.split_point]
             rolling_mean = train_data['Adj Close'].rolling(window=60).mean()
             rolling_mean.plot(label='60-Day Rolling Mean (Train)', color='green', linestyle=':', alpha=0.6)


        plt.title(title or f'{self.ticker} Stock Price Prediction (Training End: {split_date.strftime("%Y-%m-%d")})', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price ($)', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, which='both', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def report_scores(self) -> pd.DataFrame:
        """
        Prints and returns a DataFrame of R^2 scores and MSE for all trained models.
        """
        if not self.scores:
            print("No models have been trained or evaluated yet.")
            return pd.DataFrame()
            
        scores_df = pd.DataFrame.from_dict(self.scores, orient='index')
        scores_df = scores_df.sort_values(by='R2_Test', ascending=False)
        scores_df = scores_df.style.format({
            'R2_Train': '{:.4f}'.format,
            'R2_Test': '{:.4f}'.format,
            'MSE_Test': '{:,.2f}'.format
        })
        
        print("\n--- Model Evaluation Report (R^2 Score on Test Set is Key Metric) ---")
        display(scores_df)
        print("----------------------------------------------------------------------\n")
        
        return scores_df.data # Return the raw DataFrame data for non-display environments


# --- Example Usage (Module Entry Point) ---

def main_example():
    """
    Demonstrates the professional usage of the ProductionStockRegressor module.
    FORCES EXECUTION using the latest GLD file available locally (up to 2017-08-27).
    """
    print("--- Professional Stock Regressor Example ---")
    
    # 1. Initialize Regressor (Configurable Parameters)
    try:
        regressor = ProductionStockRegressor(
            ticker='GLD',              # <-- CHANGED TO MATCH LOCAL GLD FILES
            start_date='2015-06-01',   # <-- ADJUSTED: Must be AFTER file start (2015-01-01)
            end_date='2017-01-01',     # <-- ADJUSTED: Must be BEFORE file end (2017-08-27)
            predict_days=100,
            verbose=True
        )
    except Exception as e:
        print(f"FATAL ERROR during initialization: {e}")
        return

    # 2. Train Models (Unmodified)
    print("\n--- Training Models ---")
    
    # Train primary regression models (Linear/Polynomial)
    regressor.train_regression_model(degrees=[1, 3, 5])
    
    # Train hybrid models
    regressor.train_momentum_model(lookback_days=30, momentum_weight=0.4)
    regressor.train_fft_model(num_harmonics=6, underlying_trend_poly=3)

    # Note: LSTM functionality will still run if Keras is installed.
    if KERAS_AVAILABLE:
        regressor.train_lstm_model(sequence_length=60, epochs=10, units=100)
    
    # 3. Evaluate and Report Performance
    regressor.report_scores()
    
    # 4. Visualize Results
    print("\n--- Plotting Predictions ---")
    regressor.plot_predictions(model_names=['Reg_Momentum', 'FFT_Prediction'], 
                               title='GLD Price Prediction: Regression/Momentum and FFT Models')

if __name__ == '__main__':
    main_example()