"""
Data Processing for Food Supply Optimization
-------------------------------------------
Functions for loading, generating, and preprocessing data.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import requests
import json

def load_or_generate_data(base_year=2023, force_regenerate=False):
    """
    Load existing processed data or generate synthetic data.
    
    Args:
        base_year (int): Base year for climate data generation
        force_regenerate (bool): Whether to force regeneration of data
        
    Returns:
        tuple: (train_data, test_data)
    """
    train_path = f'data/processed/train_processed_full_{base_year}.csv'
    test_path = f'data/processed/test_processed_full_{base_year}.csv'
    
    if os.path.exists(train_path) and os.path.exists(test_path) and not force_regenerate:
        logging.info("Loading existing processed data files.")
        print("Loaded existing processed data files.")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    else:
        logging.info(f"Generating synthetic data for year {base_year}...")
        print(f"Generating synthetic data for year {base_year}...")
        train_data, test_data = generate_synthetic_data(base_year)
        
        # Save processed data
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
    
    return train_data, test_data

def generate_synthetic_data(base_year=2023):
    """
    Generate synthetic data for training and testing.
    
    Args:
        base_year (int): Base year for climate data generation
        
    Returns:
        tuple: (train_data, test_data)
    """
    np.random.seed(42)
    
    # Define parameters
    total_weeks = 155
    train_weeks = 145
    centers = [10, 20, 30]  # Main centers from description
    meals = [1885, 1993, 2476, 1234]  # Sample meal IDs
    
    # Create empty dataframes
    all_data = []
    
    # Generate data for each center and meal combination
    id_counter = 1
    
    for center_id in centers:
        # Map center to country and center type
        if center_id == 10:
            country = 'India'
            center_type = 'TYPE_A'
        elif center_id == 20:
            country = 'USA'
            center_type = 'TYPE_B'
        elif center_id == 30:
            country = 'UK'
            center_type = 'TYPE_C'
        else:
            country = 'Unknown'
            center_type = 'TYPE_X'
        
        for meal_id in meals:
            # Assign random category and cuisine
            category = np.random.choice(['Beverages', 'Main Course', 'Starter', 'Dessert'])
            cuisine = np.random.choice(['Indian', 'Italian', 'Continental', 'Thai', 'Chinese'])
            
            # Base parameters for this center-meal combination
            base_demand = np.random.randint(50, 150)
            base_price = np.random.randint(100, 500) / 10.0  # Generate price between 10 and 50
            
            for week in range(1, total_weeks + 1):
                # Seasonal component (yearly cycle)
                seasonal = 20 * np.sin(2 * np.pi * week / 52)
                
                # Calculate features
                # Time features
                week_sin = np.sin(2 * np.pi * week / 52)
                week_cos = np.cos(2 * np.pi * week / 52)
                
                # Price features
                discount_pct = np.random.choice([0, 0, 0, 0.1, 0.2, 0.3], p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.02])
                discount = discount_pct * base_price
                checkout_price = base_price - discount
                
                # Promotion features
                emailer = np.random.choice([0, 1], p=[0.9, 0.1])
                homepage = np.random.choice([0, 1], p=[0.8, 0.2])
                
                # Holiday indicator (random for simplicity)
                is_holiday = 1 if np.random.random() < 0.05 else 0  # 5% chance of holiday
                
                # Climate data (based on country, season and year)
                # Base temperature and precipitation with seasonal variation
                # Apply year factor (assuming slight climate change effect)
                year_factor = (base_year - 2020) * 0.05  # 0.05°C increase per year since 2020
                
                if country == 'India':
                    base_temp = 28 + 7 * np.sin(2 * np.pi * (week - 13) / 52) + year_factor  # Peak in summer (week 26)
                    base_precip = 10 + 190 * max(0, np.sin(2 * np.pi * (week - 26) / 52))  # Peak in monsoon (week 39)
                    # Adjust precipitation based on year (more extreme weather with time)
                    precip_factor = 1.0 + (base_year - 2020) * 0.02  # 2% increase in monsoon intensity per year
                    base_precip *= precip_factor
                elif country == 'USA':
                    base_temp = 15 + 15 * np.sin(2 * np.pi * (week - 13) / 52) + year_factor  # Peak in summer (week 26)
                    base_precip = 25 + 50 * max(0, np.sin(2 * np.pi * (week - 13) / 52))  # More rain in summer
                elif country == 'UK':
                    base_temp = 10 + 10 * np.sin(2 * np.pi * (week - 13) / 52) + year_factor  # Peak in summer (week 26)
                    base_precip = 30 + 20 * np.sin(2 * np.pi * week / 52)  # Fairly consistent rain
                else:
                    base_temp = 20 + 10 * np.sin(2 * np.pi * (week - 13) / 52) + year_factor  # Generic pattern
                    base_precip = 20 + 30 * np.sin(2 * np.pi * week / 52)  # Generic pattern
                
                # Add random noise to climate data
                avg_temperature = base_temp + np.random.uniform(-2, 2)
                precipitation = max(0, base_precip + np.random.uniform(-20, 20))
                
                # Calculate demand based on all factors
                demand_base = base_demand + seasonal
                
                # Promotion effects
                promo_effect = emailer * 15 + homepage * 25
                
                # Price effect (higher price reduces demand)
                price_effect = -0.5 * (checkout_price - base_price)
                
                # Holiday effect (holidays typically reduce demand)
                holiday_effect = -10 if is_holiday else 0
                
                # Weather effects
                # Optimal temperature around 25°C, demand decreases as temp deviates
                temp_effect = -0.5 * ((avg_temperature - 25) ** 2) / 10
                
                # Heavy rain reduces demand
                precip_effect = -0.05 * max(0, precipitation - 50)
                
                # Combined effect with some randomness
                num_orders = max(10, demand_base + promo_effect + price_effect + 
                               holiday_effect + temp_effect + precip_effect + 
                               np.random.normal(0, 5))
                
                # For test data (week > train_weeks), set num_orders to 0
                if week > train_weeks:
                    num_orders = 0
                
                # Create row of data
                row = {
                    'id': id_counter,
                    'week': week,
                    'center_id': center_id,
                    'meal_id': meal_id,
                    'checkout_price': checkout_price,
                    'base_price': base_price,
                    'emailer_for_promotion': emailer,
                    'homepage_featured': homepage,
                    'num_orders': num_orders,
                    'center_type': center_type,
                    'category': category,
                    'cuisine': cuisine,
                    'country': country,
                    'is_holiday': is_holiday,
                    'avg_temperature': avg_temperature,
                    'precipitation': precipitation,
                    'discount': discount,
                    'week_sin': week_sin,
                    'week_cos': week_cos
                }
                all_data.append(row)
                id_counter += 1
    
    # Convert to DataFrame
    full_data = pd.DataFrame(all_data)
    
    # Add lag features (for training data only, calculated within center-meal groups)
    for lag in range(1, 4):
        full_data[f'lag_{lag}'] = full_data.groupby(['center_id', 'meal_id'])['num_orders'].shift(lag)
    
    # Add EWMA feature
    full_data['ewma'] = full_data.groupby(['center_id', 'meal_id'])['num_orders'].transform(
        lambda x: x.ewm(span=4).mean()
    )
    
    # Split into train and test
    train_data = full_data[full_data['week'] <= train_weeks].copy()
    test_data = full_data[full_data['week'] > train_weeks].copy()
    
    # Handle missing values
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)
    
    return train_data, test_data

def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    
    Args:
        data (DataFrame): Input data
        
    Returns:
        DataFrame: Data with missing values handled
    """
    # Fill missing country values
    data['country'] = data['country'].fillna('Unknown')
    
    # Fill missing lag values with median by center_id and meal_id
    for lag in range(1, 4):
        data[f'lag_{lag}'] = data.groupby(['center_id', 'meal_id'])[f'lag_{lag}'].transform(
            lambda x: x.fillna(x.median() if not x.median() != x.median() else 0)
        )
    
    # Fill missing EWMA values
    data['ewma'] = data.groupby(['center_id', 'meal_id'])['ewma'].transform(
        lambda x: x.fillna(x.median() if not x.median() != x.median() else 0)
    )
    
    # Fill any remaining NaN values with 0
    data = data.fillna(0)
    
    return data

def preprocess_data(data, is_training=True):
    """
    Preprocess data for model training or prediction.
    
    Args:
        data (DataFrame): Input data
        is_training (bool): Whether this is for training or prediction
        
    Returns:
        tuple: (X, y) if is_training=True else X
    """
    # Select features
    features = [
        'week', 'center_id', 'meal_id', 'checkout_price', 'base_price', 
        'emailer_for_promotion', 'homepage_featured', 'center_type', 'category', 
        'cuisine', 'country', 'is_holiday', 'avg_temperature', 'precipitation', 
        'discount', 'lag_1', 'lag_2', 'lag_3', 'ewma', 'week_sin', 'week_cos'
    ]
    
    # Define expected feature columns after one-hot encoding
    # This ensures consistent features between training and prediction
    expected_columns = [
        'week', 'center_id', 'meal_id', 'checkout_price', 'base_price', 
        'emailer_for_promotion', 'homepage_featured', 'is_holiday', 
        'avg_temperature', 'precipitation', 'discount', 'lag_1', 'lag_2', 
        'lag_3', 'ewma', 'week_sin', 'week_cos',
        'center_type_TYPE_A', 'center_type_TYPE_B', 'center_type_TYPE_C',
        'category_Beverages', 'category_Dessert', 'category_Main Course', 'category_Starter',
        'cuisine_Chinese', 'cuisine_Continental', 'cuisine_Indian', 'cuisine_Italian', 'cuisine_Thai',
        'country_India', 'country_UK', 'country_USA'
    ]
    
    # Convert categorical features to one-hot encoding
    cat_features = ['center_type', 'category', 'cuisine', 'country']
    data_processed = pd.get_dummies(data[features], columns=cat_features, drop_first=False)
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in data_processed.columns:
            data_processed[col] = 0  # Add missing columns with zeros
    
    # Ensure columns are in the right order
    data_processed = data_processed[expected_columns]
    
    if is_training:
        y = data['num_orders'].values
        return data_processed, y
    else:
        return data_processed

def get_train_test_data(train_data, test_data, val_size=0.2):
    """
    Prepare train, validation, and test data.
    
    Args:
        train_data (DataFrame): Training data
        test_data (DataFrame): Test data
        val_size (float): Validation size proportion
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test)
    """
    # Get the highest week in training data
    max_week = train_data['week'].max()
    val_week_threshold = int(max_week - val_size * max_week)
    
    # Split train and validation
    train_subset = train_data[train_data['week'] <= val_week_threshold].copy()
    val_subset = train_data[train_data['week'] > val_week_threshold].copy()
    
    # Preprocess data
    X_train, y_train = preprocess_data(train_subset, is_training=True)
    X_val, y_val = preprocess_data(val_subset, is_training=True)
    X_test = preprocess_data(test_data, is_training=False)
    
    return X_train, y_train, X_val, y_val, X_test

# City to country mapping for user-friendly input
CITY_COUNTRY_MAP = {
    'mumbai': {'country': 'India', 'center_id': 10, 'center_name': 'Mumbai Center'},
    'delhi': {'country': 'India', 'center_id': 10, 'center_name': 'Delhi Center'},
    'new york': {'country': 'USA', 'center_id': 20, 'center_name': 'New York Center'},
    'chicago': {'country': 'USA', 'center_id': 20, 'center_name': 'Chicago Center'},
    'london': {'country': 'UK', 'center_id': 30, 'center_name': 'London Center'},
    'manchester': {'country': 'UK', 'center_id': 30, 'center_name': 'Manchester Center'}
}

# Meal ID to name mapping for user-friendly output
MEAL_ID_TO_NAME = {
    1885: 'Biryani',
    1993: 'Curry',
    2476: 'Pasta',
    1234: 'Salad'
}

def get_holiday_info(date, country):
    """
    Get holiday information for a specific date and country.
    
    Args:
        date (datetime): Date to check for holidays
        country (str): Country name
        
    Returns:
        dict: Holiday information with name and factor
    """
    # Default no holiday
    holiday_info = {
        'is_holiday': False,
        'holiday_name': None,
        'holiday_factor': 1.0  # Multiplier for demand (1.0 = no change)
    }
    
    # Hard-coded holidays for demo (in a real app, we'd use an API)
    holidays = {
        'India': {
            # Format: 'MM-DD': ('Holiday Name', factor)
            '01-26': ('Republic Day', 1.2),
            '08-15': ('Independence Day', 1.15),
            '10-20': ('Diwali', 1.3),  # Approximate, varies by year
            '12-25': ('Christmas', 1.1)
        },
        'USA': {
            '01-01': ('New Year', 1.1),
            '07-04': ('Independence Day', 1.2),
            '11-28': ('Thanksgiving', 1.3),  # Approximate, varies by year
            '12-25': ('Christmas', 1.25)
        },
        'UK': {
            '01-01': ('New Year', 1.1),
            '12-25': ('Christmas', 1.3),
            '12-26': ('Boxing Day', 1.15)
        }
    }
    
    # Get date in MM-DD format
    date_key = date.strftime('%m-%d')
    
    # Check if there's a holiday for this country and date
    if country in holidays and date_key in holidays[country]:
        holiday_name, factor = holidays[country][date_key]
        holiday_info['is_holiday'] = True
        holiday_info['holiday_name'] = holiday_name
        holiday_info['holiday_factor'] = factor
    
    return holiday_info

def get_weather_info(date, city):
    """
    Get weather information for a specific date and city.
    
    Args:
        date (datetime): Date to get weather for
        city (str): City name
        
    Returns:
        dict: Weather information with temperature, precipitation, and factor
    """
    # Default weather info
    weather_info = {
        'temperature': 25.0,
        'precipitation': 0.0,
        'temp_factor': 1.0,
        'precip_factor': 1.0
    }
    
    # Get base data from city mapping
    city_lower = city.lower()
    if city_lower in CITY_COUNTRY_MAP:
        country = CITY_COUNTRY_MAP[city_lower]['country']
        
        # Get week of year
        day_of_year = date.timetuple().tm_yday
        week_of_year = int(day_of_year / 7) + 1
        
        # Base year from date
        base_year = date.year
        year_factor = (base_year - 2020) * 0.05  # 0.05°C increase per year since 2020
        
        # Calculate temperature and precipitation based on country, season
        if country == 'India':
            base_temp = 28 + 7 * np.sin(2 * np.pi * (week_of_year - 13) / 52) + year_factor
            base_precip = 10 + 190 * max(0, np.sin(2 * np.pi * (week_of_year - 26) / 52))
            precip_factor = 1.0 + (base_year - 2020) * 0.02
            base_precip *= precip_factor
        elif country == 'USA':
            base_temp = 15 + 15 * np.sin(2 * np.pi * (week_of_year - 13) / 52) + year_factor
            base_precip = 25 + 50 * max(0, np.sin(2 * np.pi * (week_of_year - 13) / 52))
        elif country == 'UK':
            base_temp = 10 + 10 * np.sin(2 * np.pi * (week_of_year - 13) / 52) + year_factor
            base_precip = 30 + 20 * np.sin(2 * np.pi * week_of_year / 52)
        else:
            base_temp = 20 + 10 * np.sin(2 * np.pi * (week_of_year - 13) / 52) + year_factor
            base_precip = 20 + 30 * np.sin(2 * np.pi * week_of_year / 52)
        
        # Add random noise for more realistic weather
        temp = base_temp + np.random.uniform(-2, 2)
        precip = max(0, base_precip + np.random.uniform(-20, 20))
        
        # Calculate weather impact factors
        # Optimal temperature around 25C, demand decreases as temp deviates
        temp_factor = 1.0 - 0.01 * ((temp - 25) ** 2) / 10
        
        # Heavy rain reduces demand (>100mm is heavy rain)
        precip_factor = 1.0
        if precip > 100:
            precip_factor = 0.9  # 10% reduction for heavy rain
        elif precip > 50:
            precip_factor = 0.95  # 5% reduction for moderate rain
        
        weather_info['temperature'] = temp
        weather_info['precipitation'] = precip
        weather_info['temp_factor'] = temp_factor
        weather_info['precip_factor'] = precip_factor
    
    return weather_info

def predict_for_specific_date(date_str, city, model_year=2025, force_regenerate=False):
    """
    Generate predictions for a specific date and city.
    
    Args:
        date_str (str): Date in YYYY-MM-DD format
        city (str): City name
        model_year (int): Year to use for climate data in model
        force_regenerate (bool): Whether to force data regeneration
        
    Returns:
        tuple: (predictions, context_info)
    """
    try:
        # Parse date
        date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Please use YYYY-MM-DD format.")
    
    # Check if city is supported
    city_lower = city.lower()
    if city_lower not in CITY_COUNTRY_MAP:
        supported_cities = list(CITY_COUNTRY_MAP.keys())
        raise ValueError(f"City '{city}' not supported. Supported cities: {', '.join(supported_cities)}")
    
    # Get city and country info
    city_info = CITY_COUNTRY_MAP[city_lower]
    country = city_info['country']
    center_id = city_info['center_id']
    center_name = city_info['center_name']
    
    # Get holiday and weather information
    holiday_info = get_holiday_info(date, country)
    weather_info = get_weather_info(date, city)
    
    # Calculate week number (assuming week 1 starts on Jan 1, 2020)
    base_date = datetime(2020, 1, 1)
    days_diff = (date - base_date).days
    week = days_diff // 7 + 1
    
    # Prepare context information for output
    # Convert all values to basic Python types for JSON serialization
    # Add week reference mapping
    week_reference = {
        146: "September 16-22",
        147: "September 23-29",
        148: "September 30 - October 6",
        149: "October 7-13",
        150: "October 14-20",
        151: "October 21-27",
        152: "October 28 - November 3",
        153: "November 4-10",
        154: "November 11-17",
        155: "November 18-24"
    }

    context_info = {
        'date': date_str,
        'city': city,
        'country': country,
        'center_id': int(center_id),
        'center_name': center_name,
        'week': int(week),
        'week_reference': week_reference.get(week, "Week reference not available"),
        'holiday': {
            'is_holiday': bool(holiday_info['is_holiday']),
            'holiday_name': holiday_info['holiday_name'],
            'holiday_factor': float(holiday_info['holiday_factor'])
        },
        'weather': {
            'temperature': float(weather_info['temperature']),
            'precipitation': float(weather_info['precipitation']),
            'temp_factor': float(weather_info['temp_factor']),
            'precip_factor': float(weather_info['precip_factor'])
        }
    }
    
    # Load or generate model data
    train_data, test_data = load_or_generate_data(base_year=model_year, force_regenerate=force_regenerate)
    
    # Get processed train and test data for modeling
    X_train, y_train, X_val, y_val, X_test = get_train_test_data(train_data, test_data)
    
    # Filter test data for the specific center
    center_test_data = test_data[test_data['center_id'] == center_id].copy()
    # Process the test data for this center
    center_X_test = preprocess_data(center_test_data, is_training=False)
    
    # Load model
    from model import load_or_train_model, predict_future_demand
    lstm_model, xgb_model = load_or_train_model(X_train, y_train)
    
    # Get base predictions
    raw_predictions = predict_future_demand(lstm_model, xgb_model, center_X_test, center_test_data)
    
    # Apply holiday and weather factors to adjust demand
    adjusted_predictions = raw_predictions.copy()
    
    # Apply holiday factor
    if holiday_info['is_holiday']:
        adjusted_predictions['predicted_demand'] *= holiday_info['holiday_factor']
    
    # Apply weather factors
    adjusted_predictions['predicted_demand'] *= weather_info['temp_factor'] * weather_info['precip_factor']
    
    # Add supply prediction (demand + 10% safety buffer)
    adjusted_predictions['predicted_supply'] = adjusted_predictions['predicted_demand'] * 1.1
    
    # Add meal and center names for user-friendly output
    adjusted_predictions['meal_name'] = adjusted_predictions['meal_id'].map(MEAL_ID_TO_NAME)
    adjusted_predictions['center_name'] = center_name
    
    # Convert DataFrame to a list of dictionaries for JSON serialization
    predictions_list = []
    for _, row in adjusted_predictions.iterrows():
        prediction_dict = {
            'center_id': int(row['center_id']),
            'center_name': row['center_name'],
            'meal_id': int(row['meal_id']),
            'meal_name': row['meal_name'],
            'predicted_demand': float(row['predicted_demand']),
            'predicted_supply': float(row['predicted_supply'])
        }
        predictions_list.append(prediction_dict)
    
    return predictions_list, context_info
