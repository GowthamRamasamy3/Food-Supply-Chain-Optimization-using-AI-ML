# Food Supply Optimization Model

A comprehensive system for forecasting meal demand across multiple distribution centers and optimizing the food supply chain. This solution includes a Streamlit dashboard with interactive visualizations, machine learning models for demand forecasting, and email notification features.

## Features

- **Data Exploration**: Interactive visualization of historical demand data
- **Demand Forecasting**: Hybrid ML approach (LSTM + XGBoost) to predict future demand
- **Supply Chain Optimization**: Inventory planning with safety buffer and lead time considerations
- **Climate Impact Analysis**: Assessment of environmental impact with sustainability recommendations
- **Email Notifications**: Alert system for demand threshold monitoring

## Files and Components

- **streamlit_app.py**: Main Streamlit dashboard application
- **database.py**: PostgreSQL database connection and operations
- **email_service.py**: Email notification service
- **notification_scheduler.py**: Background scheduler for checking and sending notifications
- **test_notifications.py**: Script to test the notification system
- **model.py**: ML model implementation (LSTM + XGBoost)
- **data_processing.py**: Data preparation and preprocessing
- **climate_impact.py**: Environmental impact analysis

## Setup and Installation

1. Ensure Python 3.11+ is installed
2. Install required packages:
   ```
   pip install streamlit pandas numpy plotly matplotlib email-validator psycopg2-binary
   ```
3. Configure PostgreSQL database credentials in environment variables
4. Configure SMTP settings for email notifications:
   - SMTP_USERNAME
   - SMTP_PASSWORD
   - SMTP_SERVER (default: smtp.gmail.com)
   - SMTP_PORT (default: 587)

## Running the Application

Start the Streamlit dashboard:
```
streamlit run streamlit_app.py
```

## Usage

1. **Data Loading**: Select a base year and load historical data
2. **Model Training**: Configure model parameters and generate forecasts
3. **Supply Chain Optimization**: Set safety buffer, lead time, and cost parameters
4. **Climate Impact Analysis**: Examine environmental impact and get sustainability recommendations
5. **Email Notifications**: Register and set up demand threshold alerts

## Notification System

The application includes a background scheduler that checks demand thresholds and sends email notifications when conditions are met. To test the notification system, run:
```
python test_notifications.py
```

## Database Tables

- **users**: Store user information for notifications
- **notification_preferences**: User-defined alert thresholds
- **notifications**: History of sent notifications