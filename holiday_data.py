"""
Holiday and Special Event Data for Food Supply Optimization
---------------------------------------------------------
Data about holidays, festivals, and special events that affect demand.
"""

# Define global holiday calendar
# Format: {(month, day): {'name': 'Holiday Name', 'demand_factor': float}}
# demand_factor: multiplier for expected demand (>1 means higher demand)
GLOBAL_HOLIDAYS = {
    (1, 1): {'name': 'New Year\'s Day', 'demand_factor': 1.2},
    (12, 25): {'name': 'Christmas Day', 'demand_factor': 1.5},
    (12, 31): {'name': 'New Year\'s Eve', 'demand_factor': 1.4},
}

# Country-specific holidays
COUNTRY_HOLIDAYS = {
    'India': {
        # Deepavali/Diwali (date varies by year)
        (11, 12): {'name': 'Deepavali/Diwali (2025)', 'demand_factor': 1.8},
        (10, 31): {'name': 'Deepavali/Diwali (2024)', 'demand_factor': 1.8},
        (11, 1): {'name': 'Deepavali (2023)', 'demand_factor': 1.8},
        # Other Indian holidays
        (8, 15): {'name': 'Independence Day', 'demand_factor': 1.3},
        (1, 26): {'name': 'Republic Day', 'demand_factor': 1.3},
        (10, 2): {'name': 'Gandhi Jayanti', 'demand_factor': 1.2},
        (3, 18): {'name': 'Holi (2025)', 'demand_factor': 1.7},
        (8, 26): {'name': 'Raksha Bandhan (2025)', 'demand_factor': 1.4},
        (9, 3): {'name': 'Ganesh Chaturthi (2025)', 'demand_factor': 1.5},
    },
    'USA': {
        (7, 4): {'name': 'Independence Day', 'demand_factor': 1.5},
        (11, 28): {'name': 'Thanksgiving (2025)', 'demand_factor': 1.7},
        (11, 27): {'name': 'Thanksgiving (2024)', 'demand_factor': 1.7},
        (11, 23): {'name': 'Thanksgiving (2023)', 'demand_factor': 1.7},
        (2, 14): {'name': 'Valentine\'s Day', 'demand_factor': 1.6},
        (5, 26): {'name': 'Memorial Day (2025)', 'demand_factor': 1.3},
        (5, 27): {'name': 'Memorial Day (2024)', 'demand_factor': 1.3},
        (5, 29): {'name': 'Memorial Day (2023)', 'demand_factor': 1.3},
    },
    'China': {
        (2, 1): {'name': 'Chinese New Year (2025)', 'demand_factor': 1.9},
        (1, 22): {'name': 'Chinese New Year (2024)', 'demand_factor': 1.9},
        (1, 22): {'name': 'Chinese New Year (2023)', 'demand_factor': 1.9},
        (10, 1): {'name': 'National Day', 'demand_factor': 1.4},
        (5, 1): {'name': 'Labour Day', 'demand_factor': 1.3},
        (9, 15): {'name': 'Mid-Autumn Festival (2025)', 'demand_factor': 1.5},
    },
    'United Kingdom': {
        (12, 26): {'name': 'Boxing Day', 'demand_factor': 1.3},
        (5, 5): {'name': 'Bank Holiday (2025)', 'demand_factor': 1.2},
        (5, 6): {'name': 'Bank Holiday (2024)', 'demand_factor': 1.2},
        (5, 1): {'name': 'Bank Holiday (2023)', 'demand_factor': 1.2},
    }
}

# Special events that affect demand
SPECIAL_EVENTS = {
    2025: [
        {'name': 'FIFA World Cup Final', 'date': (7, 13), 'demand_factor': 1.6, 'countries': ['global']},
        {'name': 'Super Bowl', 'date': (2, 9), 'demand_factor': 1.8, 'countries': ['USA']},
        {'name': 'Cricket World Cup Final', 'date': (11, 23), 'demand_factor': 1.7, 'countries': ['India', 'United Kingdom', 'Australia']},
    ],
    2024: [
        {'name': 'Summer Olympics', 'date': (7, 26), 'demand_factor': 1.5, 'countries': ['global']},
        {'name': 'Super Bowl', 'date': (2, 11), 'demand_factor': 1.8, 'countries': ['USA']},
        {'name': 'UEFA Euro Final', 'date': (7, 14), 'demand_factor': 1.6, 'countries': ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy']},
    ],
    2023: [
        {'name': 'Super Bowl', 'date': (2, 12), 'demand_factor': 1.8, 'countries': ['USA']},
        {'name': 'Cricket World Cup Final', 'date': (11, 19), 'demand_factor': 1.7, 'countries': ['India', 'United Kingdom', 'Australia']},
    ]
}

def get_holiday_info(date, country='global'):
    """
    Get holiday information for a specific date and country.
    
    Args:
        date (datetime.date): Date to check for holidays
        country (str): Country name or 'global' for global holidays
        
    Returns:
        dict: Holiday information with name and demand factor
    """
    month, day = date.month, date.day
    year = date.year
    
    # Check for global holidays
    holiday_info = GLOBAL_HOLIDAYS.get((month, day), None)
    
    # Check for country-specific holidays
    if country != 'global' and country in COUNTRY_HOLIDAYS:
        country_holiday = COUNTRY_HOLIDAYS[country].get((month, day), None)
        if country_holiday:
            holiday_info = country_holiday
    
    # Check for special events
    if year in SPECIAL_EVENTS:
        for event in SPECIAL_EVENTS[year]:
            if event['date'] == (month, day) and (event['countries'][0] == 'global' or country in event['countries']):
                holiday_info = event
                break
    
    # Return default if no holiday found
    if not holiday_info:
        return {'name': 'No Holiday', 'demand_factor': 1.0}
    
    return holiday_info

def get_weather_impact(temperature, precipitation):
    """
    Calculate the impact of weather on demand.
    
    Args:
        temperature (float): Temperature in Celsius
        precipitation (float): Precipitation in mm
        
    Returns:
        float: Weather impact factor for demand
    """
    # Base impact
    impact = 1.0
    
    # Temperature impact (higher in extreme temperatures)
    if temperature < 10:
        # Cold weather increases demand
        temp_impact = 1.0 + (10 - temperature) * 0.02  # +2% per degree below 10°C
    elif temperature > 30:
        # Hot weather for hot meals decreases demand
        temp_impact = 1.0 - (temperature - 30) * 0.03  # -3% per degree above 30°C
    else:
        # Moderate temperature
        temp_impact = 1.0
    
    # Precipitation impact (rain/snow increases demand)
    if precipitation > 0:
        precip_impact = 1.0 + min(precipitation * 0.01, 0.3)  # +1% per mm, max +30%
    else:
        precip_impact = 1.0
    
    # Combine impacts
    impact = temp_impact * precip_impact
    
    return impact

# Synthetic monthly average temperature data for major cities (°C)
CITY_TEMPERATURES = {
    'New York': {1: 0, 2: 1, 3: 6, 4: 12, 5: 18, 6: 23, 7: 26, 8: 25, 9: 21, 10: 15, 11: 9, 12: 3},
    'London': {1: 5, 2: 5, 3: 8, 4: 11, 5: 14, 6: 17, 7: 19, 8: 19, 9: 16, 10: 12, 11: 8, 12: 6},
    'Tokyo': {1: 6, 2: 6, 3: 9, 4: 14, 5: 19, 6: 22, 7: 26, 8: 27, 9: 23, 10: 18, 11: 13, 12: 8},
    'Mumbai': {1: 24, 2: 25, 3: 28, 4: 30, 5: 32, 6: 30, 7: 28, 8: 28, 9: 28, 10: 29, 11: 28, 12: 26},
    'Sydney': {1: 23, 2: 23, 3: 22, 4: 19, 5: 16, 6: 14, 7: 13, 8: 14, 9: 16, 10: 18, 11: 20, 12: 22},
    'Paris': {1: 5, 2: 6, 3: 9, 4: 12, 5: 16, 6: 19, 7: 21, 8: 21, 9: 18, 10: 13, 11: 9, 12: 6},
    'Beijing': {1: -4, 2: 0, 3: 7, 4: 14, 5: 20, 6: 25, 7: 26, 8: 25, 9: 20, 10: 13, 11: 5, 12: -2},
    'Berlin': {1: 0, 2: 1, 3: 5, 4: 10, 5: 15, 6: 18, 7: 20, 8: 19, 9: 15, 10: 10, 11: 5, 12: 2},
    'Mexico City': {1: 14, 2: 16, 3: 18, 4: 19, 5: 20, 6: 19, 7: 18, 8: 18, 9: 18, 10: 17, 11: 16, 12: 14},
    'Cairo': {1: 14, 2: 15, 3: 18, 4: 22, 5: 26, 6: 28, 7: 29, 8: 29, 9: 27, 10: 25, 11: 20, 12: 16},
}

# Synthetic monthly average precipitation data for major cities (mm per day)
CITY_PRECIPITATION = {
    'New York': {1: 3, 2: 3, 3: 4, 4: 4, 5: 4, 6: 3, 7: 4, 8: 4, 9: 3, 10: 3, 11: 4, 12: 3},
    'London': {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 2, 12: 2},
    'Tokyo': {1: 2, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 5, 8: 5, 9: 6, 10: 5, 11: 4, 12: 2},
    'Mumbai': {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 15, 7: 20, 8: 17, 9: 10, 10: 3, 11: 1, 12: 0},
    'Sydney': {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 5, 7: 3, 8: 3, 9: 3, 10: 3, 11: 4, 12: 3},
    'Paris': {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2},
    'Beijing': {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 8, 7: 15, 8: 12, 9: 5, 10: 2, 11: 1, 12: 0},
    'Berlin': {1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 1, 11: 2, 12: 2},
    'Mexico City': {1: 0, 2: 0, 3: 1, 4: 1, 5: 3, 6: 5, 7: 5, 8: 5, 9: 4, 10: 2, 11: 1, 12: 0},
    'Cairo': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
}

def get_city_climate_data(date, city):
    """
    Get climate data for a specific date and city.
    
    Args:
        date (datetime.date): Date to get climate data for
        city (str): City name
        
    Returns:
        dict: Climate data including temperature and precipitation
    """
    month = date.month
    
    # Default values if city not found
    temp = 20.0
    precip = 0.0
    
    # Get temperature and precipitation from our data
    if city in CITY_TEMPERATURES:
        temp = CITY_TEMPERATURES[city][month]
        
    if city in CITY_PRECIPITATION:
        precip = CITY_PRECIPITATION[city][month]
    
    # Add some daily variation (±3°C, ±2mm)
    import random
    random.seed(date.day + date.month * 31 + date.year * 366)
    temp_variation = random.uniform(-3, 3)
    precip_variation = random.uniform(-2, 2) if precip > 0 else random.uniform(0, 2)
    
    # Ensure precipitation is not negative
    actual_precip = max(0, precip + precip_variation)
    actual_temp = temp + temp_variation
    
    # Calculate demand impact
    weather_impact = get_weather_impact(actual_temp, actual_precip)
    
    return {
        'temperature': round(actual_temp, 1),
        'precipitation': round(actual_precip, 1),
        'weather_impact': round(weather_impact, 2)
    }

def get_combined_factors(date, city, country='global'):
    """
    Get combined demand factors for a specific date, city, and country.
    
    Args:
        date (datetime.date): Date to analyze
        city (str): City name
        country (str): Country name
        
    Returns:
        dict: Combined factors affecting demand
    """
    # Get holiday information
    holiday_info = get_holiday_info(date, country)
    
    # Get climate data
    climate_data = get_city_climate_data(date, city)
    
    # Calculate weekday factor (weekends have higher demand)
    weekday = date.weekday()  # 0-6 (Monday-Sunday)
    weekday_factor = 1.2 if weekday >= 5 else 1.0  # Weekend boost
    
    # Calculate combined factor
    combined_factor = holiday_info['demand_factor'] * climate_data['weather_impact'] * weekday_factor
    
    return {
        'holiday': holiday_info['name'],
        'holiday_factor': holiday_info['demand_factor'],
        'temperature': climate_data['temperature'],
        'precipitation': climate_data['precipitation'],
        'weather_factor': climate_data['weather_impact'],
        'weekday_factor': weekday_factor,
        'combined_factor': round(combined_factor, 2)
    }