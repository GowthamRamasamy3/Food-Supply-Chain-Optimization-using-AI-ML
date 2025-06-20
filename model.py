"""
Model Definition and Training for Food Supply Optimization
---------------------------------------------------------
Functions for LSTM and XGBoost modeling and prediction.
"""
import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math

def root_mean_squared_log_error(y_true, y_pred):
    """
    Calculate RMSLE (Root Mean Squared Logarithmic Error)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        float: RMSLE value
    """
    # Ensure no negative predictions
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)
    
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def prepare_lstm_data(X, y=None, sequence_length=4):
    """
    Prepare data for LSTM model with sequence structure.
    
    Args:
        X (DataFrame): Features data
        y (array): Target values, optional
        sequence_length (int): Length of sequences to create
        
    Returns:
        tuple: (X_lstm, y_lstm) if y is provided, else X_lstm
    """
    # Get unique center-meal combinations
    if y is not None:
        # For training data
        data = X.copy()
        data['num_orders'] = y
        
        sequences_X = []
        sequences_y = []
        
        # Group by center_id and meal_id
        for name, group in data.groupby(['center_id', 'meal_id']):
            # Sort by week to ensure proper sequence
            group = group.sort_values('week')
            
            # Extract features and target
            group_X = group.drop(['center_id', 'meal_id', 'num_orders'], axis=1).values
            group_y = group['num_orders'].values
            
            # Create sequences
            for i in range(len(group) - sequence_length):
                sequences_X.append(group_X[i:i+sequence_length])
                sequences_y.append(group_y[i+sequence_length])
        
        if not sequences_X:
            # If there are no sequences (e.g., not enough data), return empty arrays
            return np.array([]), np.array([])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    else:
        # For prediction data
        sequences_X = []
        
        # Group by center_id and meal_id
        for name, group in X.groupby(['center_id', 'meal_id']):
            # Sort by week to ensure proper sequence
            group = group.sort_values('week')
            
            # Extract features
            group_X = group.drop(['center_id', 'meal_id'], axis=1).values
            
            # Use the last sequence_length records for prediction
            if len(group) >= sequence_length:
                sequences_X.append(group_X[-sequence_length:])
        
        if not sequences_X:
            # If there are no sequences (e.g., not enough data), return empty array
            return np.array([])
        
        return np.array(sequences_X)

def build_lstm_model(input_shape):
    """
    Build an LSTM model for time series forecasting.
    
    Args:
        input_shape (tuple): Shape of input data
        
    Returns:
        Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def load_or_train_model(X_train, y_train):
    """
    Load existing models or train new ones if they don't exist.
    
    Args:
        X_train (DataFrame): Training features
        y_train (array): Training targets
        
    Returns:
        tuple: (lstm_model, xgb_model)
    """
    lstm_model_path = 'models/full/hybrid_lstm_model.keras'
    xgb_model_path = 'models/full/hybrid_xgb_model.pkl'
    
    try:
        if os.path.exists(lstm_model_path) and os.path.exists(xgb_model_path):
            logging.info("Loading existing models.")
            print("Loaded existing model from pickle files.")
            
            lstm_model = load_model(lstm_model_path)
            
            with open(xgb_model_path, 'rb') as file:
                xgb_model = pickle.load(file)
            
            return lstm_model, xgb_model
    except Exception as e:
        logging.error(f"Error loading existing models: {str(e)}")
        print(f"Error loading existing models: {str(e)}")
    
    logging.info("No existing model found or loading failed. Training new model...")
    print("No existing model found or loading failed. Training new model...")
    
    # Create a copy of X_train for XGBoost model
    X_train_xgb = X_train.copy()
    
    # Prepare data for LSTM
    X_lstm, y_lstm = prepare_lstm_data(X_train, y_train)
    
    # Handle case where there's not enough sequential data
    if len(X_lstm) == 0:
        logging.warning("Not enough sequential data for LSTM model. Using only XGBoost.")
        lstm_model = None
    else:
        # Build and train LSTM model
        input_shape = (X_lstm.shape[1], X_lstm.shape[2])
        lstm_model = build_lstm_model(input_shape)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Convert to correct dtype for TensorFlow
        X_lstm = tf.convert_to_tensor(X_lstm, dtype=tf.float32)
        y_lstm = tf.convert_to_tensor(y_lstm, dtype=tf.float32)
        
        lstm_model.fit(
            X_lstm, y_lstm,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save LSTM model
        lstm_model.save(lstm_model_path)
    
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    xgb_model.fit(X_train_xgb, y_train)
    
    # Save XGBoost model
    with open(xgb_model_path, 'wb') as file:
        pickle.dump(xgb_model, file)
    
    return lstm_model, xgb_model

def evaluate_model(lstm_model, xgb_model, X_val, y_val):
    """
    Evaluate the hybrid model on validation data.
    
    Args:
        lstm_model: LSTM model
        xgb_model: XGBoost model
        X_val (DataFrame): Validation features
        y_val (array): Validation targets
        
    Returns:
        array: Validation predictions
    """
    # Make predictions with XGBoost
    xgb_preds = xgb_model.predict(X_val)
    
    # Make predictions with LSTM if available
    if lstm_model is not None:
        X_val_lstm = prepare_lstm_data(X_val)
        if len(X_val_lstm) > 0:
            # Convert to correct dtype for TensorFlow
            X_val_lstm = tf.convert_to_tensor(X_val_lstm, dtype=tf.float32)
            lstm_preds = lstm_model.predict(X_val_lstm).flatten()
            
            # Match LSTM predictions to original data points
            # This is a simplification - in a real scenario, you would need to match sequences to rows
            if len(lstm_preds) < len(xgb_preds):
                # Pad with XGBoost predictions
                padded_lstm_preds = np.zeros_like(xgb_preds)
                padded_lstm_preds[-len(lstm_preds):] = lstm_preds
                lstm_preds = padded_lstm_preds
            elif len(lstm_preds) > len(xgb_preds):
                # Truncate LSTM predictions
                lstm_preds = lstm_preds[-len(xgb_preds):]
            
            # Ensemble predictions (simple average)
            val_preds = (lstm_preds + xgb_preds) / 2
        else:
            val_preds = xgb_preds
    else:
        val_preds = xgb_preds
    
    # Ensure non-negative predictions
    val_preds = np.maximum(val_preds, 0)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    rmsle = root_mean_squared_log_error(y_val, val_preds)
    mae = mean_absolute_error(y_val, val_preds)
    
    # Calculate MAPE, handling zeros in y_val
    y_val_nonzero = np.where(y_val > 0, y_val, 1e-10)  # Replace zeros with small value
    mape = mean_absolute_percentage_error(y_val_nonzero, val_preds)
    
    # Calculate accuracy (based on predictions within 20% of actual values)
    accuracy = np.mean(np.abs(val_preds - y_val) / y_val_nonzero <= 0.2) * 100
    
    print("Validation Metrics:")
    print(f"RMSLE: {rmsle:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape * 100:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return val_preds

def predict_future_demand(lstm_model, xgb_model, X_test, test_data):
    """
    Predict future demand using the hybrid model.
    
    Args:
        lstm_model: LSTM model
        xgb_model: XGBoost model
        X_test (DataFrame): Test features
        test_data (DataFrame): Original test data
        
    Returns:
        DataFrame: Predictions in the required format
    """
    # Make predictions with XGBoost
    xgb_preds = xgb_model.predict(X_test)
    
    # Make predictions with LSTM if available
    if lstm_model is not None:
        X_test_lstm = prepare_lstm_data(X_test)
        if len(X_test_lstm) > 0:
            # Convert to correct dtype for TensorFlow
            X_test_lstm = tf.convert_to_tensor(X_test_lstm, dtype=tf.float32)
            lstm_preds = lstm_model.predict(X_test_lstm).flatten()
            
            # Match LSTM predictions to original data points
            if len(lstm_preds) < len(xgb_preds):
                # Create an empty array filled with XGBoost predictions as fallback
                combined_preds = xgb_preds.copy()
                
                # Map LSTM predictions to their corresponding positions
                # This is a simplification, assuming the last sequences correspond to test data
                combined_preds[-len(lstm_preds):] = (lstm_preds + xgb_preds[-len(lstm_preds):]) / 2
                test_preds = combined_preds
            elif len(lstm_preds) > len(xgb_preds):
                # Truncate LSTM predictions
                lstm_preds = lstm_preds[-len(xgb_preds):]
                test_preds = (lstm_preds + xgb_preds) / 2
            else:
                # Equal length, simple average
                test_preds = (lstm_preds + xgb_preds) / 2
        else:
            test_preds = xgb_preds
    else:
        test_preds = xgb_preds
    
    # Ensure non-negative predictions
    test_preds = np.maximum(test_preds, 0)
    
    # Create prediction DataFrame
    predictions = pd.DataFrame({
        'week': test_data['week'],
        'center_id': test_data['center_id'],
        'meal_id': test_data['meal_id'],
        'predicted_demand': test_preds,
        'date': pd.date_range(start=datetime.now(), periods=len(test_preds), freq='W-MON')
    })
    
    # Convert DataFrame to a list of dictionaries for JSON serialization
    if 'to_dict' in predictions:
        return predictions
    else:
        predictions_list = []
        for _, row in predictions.iterrows():
            prediction_dict = {
                'week': int(row['week']),
                'center_id': int(row['center_id']),
                'meal_id': int(row['meal_id']),
                'predicted_demand': float(row['predicted_demand'])
            }
            predictions_list.append(prediction_dict)
        return predictions_list
