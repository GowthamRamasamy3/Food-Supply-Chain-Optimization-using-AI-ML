import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataProcessor:
    """
    Class for processing meal demand data for forecasting.
    Handles data cleaning, feature engineering, and preparation for model training.
    """
    
    def __init__(self):
        """Initialize the DataProcessor with necessary transformers."""
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_names = None
        self.sequence_length = None
    
    def preprocess_data(self, data):
        """
        Preprocess the raw data, including cleaning and feature engineering.
        
        Args:
            data (pd.DataFrame): Raw meal demand data
            
        Returns:
            pd.DataFrame: Processed data ready for model input
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check for missing values in critical columns
        critical_cols = ['week', 'center_id', 'meal_id', 'num_orders']
        if df[critical_cols].isnull().sum().sum() > 0:
            # Handle missing values in critical columns
            for col in critical_cols:
                if df[col].isnull().sum() > 0:
                    if col in ['center_id', 'meal_id']:
                        # For categorical identifiers, we can't impute
                        df = df.dropna(subset=[col])
                    else:
                        # For numerical values, use median imputation
                        df[col] = df[col].fillna(df[col].median())
        
        # Convert data types
        if 'week' in df.columns:
            df['week'] = df['week'].astype(int)
        if 'center_id' in df.columns:
            df['center_id'] = df['center_id'].astype(int)
        if 'meal_id' in df.columns:
            df['meal_id'] = df['meal_id'].astype(int)
        
        # Feature engineering
        
        # 1. Add time-based features (if not already present)
        if 'week_of_year' not in df.columns and 'week' in df.columns:
            df['week_of_year'] = df['week'] % 52
            df['quarter'] = (df['week_of_year'] // 13) + 1
        
        # 2. Add lagged features for demand
        if 'num_orders' in df.columns:
            # Add lags of 1, 2, and 4 weeks
            for lag in [1, 2, 4]:
                df[f'lag_{lag}_demand'] = df.groupby(['center_id', 'meal_id'])['num_orders'].shift(lag)
            
            # Add rolling statistics
            for window in [4, 8]:
                # Rolling mean
                df[f'rolling_{window}_mean'] = df.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean())
                
                # Rolling std
                df[f'rolling_{window}_std'] = df.groupby(['center_id', 'meal_id'])['num_orders'].transform(
                    lambda x: x.rolling(window, min_periods=1).std())
        
        # 3. Handle holidays and promotional periods if present
        if 'is_holiday' in df.columns:
            df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        
        if 'promotion_type' in df.columns:
            # One-hot encode promotion type
            df = pd.get_dummies(df, columns=['promotion_type'], prefix='promo')
        
        # 4. Add center and meal specific features
        # Calculate average and std demand per center and meal
        if 'num_orders' in df.columns:
            center_stats = df.groupby('center_id')['num_orders'].agg(['mean', 'std']).reset_index()
            center_stats.columns = ['center_id', 'center_avg_demand', 'center_std_demand']
            
            meal_stats = df.groupby('meal_id')['num_orders'].agg(['mean', 'std']).reset_index()
            meal_stats.columns = ['meal_id', 'meal_avg_demand', 'meal_std_demand']
            
            # Merge back to the main dataframe
            df = pd.merge(df, center_stats, on='center_id', how='left')
            df = pd.merge(df, meal_stats, on='meal_id', how='left')
        
        # 5. Handle any remaining missing values
        # For numerical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
        
        # For categorical columns (if any)
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        return df
    
    def prepare_training_data(self, df, sequence_length=12, test_size=0.2):
        """
        Prepare data for training the hybrid model.
        
        Args:
            df (pd.DataFrame): Preprocessed data
            sequence_length (int): Length of sequence for LSTM
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        # Store sequence length for later use
        self.sequence_length = sequence_length
        
        # Ensure data is sorted by week for each center and meal
        df = df.sort_values(by=['center_id', 'meal_id', 'week'])
        
        # Features and target
        target_col = 'num_orders'
        
        # Exclude non-feature columns and the target
        exclude_cols = ['num_orders', 'week', 'center_id', 'meal_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        # Create sequences for LSTM
        sequences = []
        targets = []
        
        # Group by center and meal
        for (center, meal), group in df.groupby(['center_id', 'meal_id']):
            if len(group) < sequence_length + 1:
                continue
                
            # Create sequences
            for i in range(len(group) - sequence_length):
                sequences.append(group[feature_cols].iloc[i:i+sequence_length].values)
                targets.append(group[target_col].iloc[i+sequence_length])
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets).reshape(-1, 1)
        
        # Split into train and test sets
        indices = np.arange(len(X))
        X_train_idx, X_test_idx, y_train, y_test = train_test_split(
            indices, y, test_size=test_size, random_state=42
        )
        
        X_train = X[X_train_idx]
        X_test = X[X_test_idx]
        
        # Scale numerical features
        # We need to reshape the data to apply scaling
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        
        X_train_scaled = self.numerical_scaler.fit_transform(X_train_reshaped)
        X_train = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
        
        n_samples_test = X_test.shape[0]
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = self.numerical_scaler.transform(X_test_reshaped)
        X_test = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def prepare_forecast_input(self, data, center_id, meal_id, sequence_length=None):
        """
        Prepare input data for forecasting for a specific center and meal.
        
        Args:
            data (pd.DataFrame): Processed data
            center_id (int): ID of the center
            meal_id (int): ID of the meal
            sequence_length (int, optional): Length of sequence. If None, uses the one from training.
            
        Returns:
            numpy.ndarray: Prepared input for the model
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        if sequence_length is None:
            raise ValueError("Sequence length must be provided if not set during training")
        
        # Filter data for the specific center and meal
        filtered_data = data[(data['center_id'] == center_id) & (data['meal_id'] == meal_id)]
        
        # Sort by week
        filtered_data = filtered_data.sort_values(by='week')
        
        # Check if we have enough data
        if len(filtered_data) < sequence_length:
            return None
        
        # Get the most recent sequence
        recent_data = filtered_data.iloc[-sequence_length:]
        
        # Select feature columns
        feature_data = recent_data[self.feature_names].values
        
        # Apply scaling
        feature_data_reshaped = feature_data.reshape(-1, len(self.feature_names))
        feature_data_scaled = self.numerical_scaler.transform(feature_data_reshaped)
        feature_data = feature_data_scaled.reshape(1, sequence_length, len(self.feature_names))
        
        return feature_data
