"""
Food Supply Optimization Model - Streamlit Dashboard
---------------------------------------------------
Interactive Streamlit dashboard for food demand forecasting and supply chain optimization.
"""

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import random
import logging
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import notification scheduler
try:
    from notification_scheduler import start_scheduler_thread
    # Start notification scheduler in the background (check every 30 minutes)
    if 'notification_thread' not in st.session_state:
        st.session_state.notification_thread = start_scheduler_thread(interval_minutes=30)
        logger.info("Started notification scheduler in the background")
except Exception as e:
    logger.error(f"Failed to start notification scheduler: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Food Demand Forecasting & Supply Chain Optimization",
    page_icon="🍲",
    layout="wide"
)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'predictions_generated' not in st.session_state:
    st.session_state.predictions_generated = False
if 'supply_chain_optimized' not in st.session_state:
    st.session_state.supply_chain_optimized = False

# Title and description
st.title('🍲 Food Demand Forecasting & Supply Chain Optimization')
st.markdown("""
This application uses hybrid machine learning models (LSTM + XGBoost) to forecast meal demand
across multiple centers and optimize supply chain operations.
""")

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select a page', [
    'Data Loading & Exploration', 
    'Model Training & Forecasting', 
    'Supply Chain Optimization', 
    'Climate Impact Analysis',
    'Report Generation',
    'Email Notifications'
])

# Load data function
@st.cache_data
def load_data(year='2023'):
    """Load training and test data"""
    try:
        data_dir = os.path.join(os.getcwd(), 'data', 'processed')
        train_file = os.path.join(data_dir, f'train_processed_full_{year}.csv')
        test_file = os.path.join(data_dir, f'test_processed_full_{year}.csv')
        
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        
        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load predictions function
@st.cache_data
def load_predictions(year='2023'):
    """Load prediction results"""
    try:
        results_dir = os.path.join(os.getcwd(), 'results', 'full')
        predictions_file = os.path.join(results_dir, f'food_supply_optimization_predictions_{year}.csv')
        
        if os.path.exists(predictions_file):
            predictions = pd.read_csv(predictions_file)
            return predictions
        else:
            return None
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return None

# Data Loading & Exploration page
if page == 'Data Loading & Exploration':
    st.header('Data Loading & Exploration')
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_year = st.selectbox('Select base year for data', ['2023', '2025'], index=0)
        
    with col2:
        force_regenerate = st.checkbox('Force regenerate data', value=False)
    
    if st.button('Load Data'):
        with st.spinner('Loading data, please wait...'):
            try:
                # Load data
                train_data, test_data = load_data(base_year)
                
                if train_data is not None and test_data is not None:
                    # Store in session state
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    st.session_state.data_loaded = True
                    
                    st.success(f'Data loaded successfully! Base year: {base_year}')
                else:
                    st.error('Failed to load data.')
            except Exception as e:
                st.error(f'Error loading data: {str(e)}')
    
    # Data exploration section
    if st.session_state.data_loaded:
        st.subheader('Data Overview')
        
        train_data = st.session_state.train_data
        test_data = st.session_state.test_data
        
        # Basic statistics
        st.write(f"Training data shape: {train_data.shape}")
        st.write(f"Test data shape: {test_data.shape}")
        
        # Display data samples
        tabs = st.tabs(['Training Data', 'Test Data', 'Visualizations'])
        
        with tabs[0]:
            st.dataframe(train_data.head(10))
            st.write("Statistical Summary:")
            st.dataframe(train_data.describe())
        
        with tabs[1]:
            st.dataframe(test_data.head(10))
            st.write("Statistical Summary:")
            st.dataframe(test_data.describe())
        
        with tabs[2]:
            # Time series visualization
            st.subheader('Meal Demand Over Time')
            
            # Filters for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                center_ids = sorted(train_data['center_id'].unique())
                selected_center = st.selectbox('Select Center ID', center_ids)
            
            with col2:
                meal_ids = sorted(train_data['meal_id'].unique())
                selected_meal = st.selectbox('Select Meal ID', meal_ids)
            
            # Filter data
            filtered_data = train_data[
                (train_data['center_id'] == selected_center) & 
                (train_data['meal_id'] == selected_meal)
            ]
            
            # Plot time series
            if not filtered_data.empty:
                fig = px.line(
                    filtered_data, 
                    x='week', 
                    y='num_orders',
                    title=f'Demand Pattern for Center {selected_center}, Meal {selected_meal}'
                )
                fig.update_layout(
                    xaxis_title='Week',
                    yaxis_title='Number of Orders',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Price vs Demand
                if 'checkout_price' in filtered_data.columns:
                    st.subheader('Price vs Demand')
                    price_demand_fig = px.scatter(
                        filtered_data,
                        x='checkout_price',
                        y='num_orders',
                        title='Price vs Demand Relationship'
                    )
                    price_demand_fig.update_layout(
                        xaxis_title='Checkout Price',
                        yaxis_title='Number of Orders',
                        height=400
                    )
                    st.plotly_chart(price_demand_fig, use_container_width=True)
            else:
                st.warning("No data available for the selected center and meal combination.")

# Model Training & Forecasting page
elif page == 'Model Training & Forecasting':
    st.header('Model Training & Forecasting')
    
    # Create tabs for different forecasting approaches
    train_tab, load_tab, date_tab = st.tabs([
        "Train Model & Forecast", 
        "Load Existing Forecasts",
        "Date-Specific Forecasting"
    ])
    
    # Train Model & Forecast tab
    with train_tab:
        if not st.session_state.data_loaded:
            st.warning('Please load data first from the Data Loading & Exploration page.')
        else:
            st.subheader('Model Configuration')
            
            # Model parameters
            col1, col2 = st.columns(2)
            
            with col1:
                base_year = st.selectbox('Select base year for model', ['2023', '2025'], index=0, key="train_base_year")
                lstm_units = st.slider('LSTM Units', min_value=32, max_value=256, value=128, step=32)
                lstm_dropout = st.slider('LSTM Dropout Rate', min_value=0.1, max_value=0.5, value=0.2, step=0.1)
            
            with col2:
                xgb_max_depth = st.slider('XGBoost Max Depth', min_value=3, max_value=10, value=6, step=1)
                xgb_learning_rate = st.slider('XGBoost Learning Rate', min_value=0.01, max_value=0.3, value=0.1, step=0.05)
                forecast_horizon = st.slider('Forecast Horizon (weeks)', min_value=1, max_value=12, value=4, step=1)
            
            # Train/load model button
            if st.button('Train/Load Models', key="train_tab_button"):
                with st.spinner('Training/loading models, please wait...'):
                    # Simulating model training with a progress bar
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(percent_complete + 1)
                    
                    try:
                        # Try to load predictions directly
                        predictions = load_predictions(base_year)
                        
                        if predictions is not None:
                            st.session_state.predictions = predictions
                            st.session_state.models_loaded = True
                            st.session_state.predictions_generated = True
                            
                            st.success('Models loaded and predictions generated successfully!')
                        else:
                            st.error('Failed to load predictions. Please check if prediction files exist.')
                    except Exception as e:
                        st.error(f'Error loading models and predictions: {str(e)}')
    
    # Load Existing Forecasts tab
    with load_tab:
        st.subheader('Load Existing Forecasts')
        
        base_year = st.selectbox('Select year for predictions', ['2023', '2024', '2025'], index=0, key="load_base_year")
        
        # Load models button
        if st.button('Load Predictions', key="load_tab_button"):
            with st.spinner('Loading predictions, please wait...'):
                # Simulating data loading with a progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(percent_complete + 1)
                
                try:
                    # Try to load predictions directly
                    predictions = load_predictions(base_year)
                    
                    if predictions is not None:
                        st.session_state.predictions = predictions
                        st.session_state.models_loaded = True
                        st.session_state.predictions_generated = True
                        
                        st.success('Predictions loaded successfully!')
                    else:
                        st.error('Failed to load predictions. Please check if prediction files exist.')
                except Exception as e:
                    st.error(f'Error loading predictions: {str(e)}')
        
        # Display forecast results
        if st.session_state.predictions_generated and 'predictions' in st.session_state:
            st.subheader('Forecast Results')
            
            predictions = st.session_state.predictions
            
            # Show tabular results
            st.dataframe(predictions.head(20))
            
            # Download option
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"food_demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Visualization of forecast
            st.subheader('Forecast Visualization')
            
            # Filters for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                center_ids = sorted(predictions['center_id'].unique())
                selected_center = st.selectbox('Select Center ID for Visualization', center_ids, key="load_center_viz")
            
            with col2:
                meal_ids = sorted(predictions['meal_id'].unique())
                selected_meal = st.selectbox('Select Meal ID for Visualization', meal_ids, key="load_meal_viz")
            
            # Filter predictions
            center_meal_forecast = predictions[
                (predictions['center_id'] == selected_center) &
                (predictions['meal_id'] == selected_meal)
            ]
            
            # Plot forecast
            if not center_meal_forecast.empty:
                # Determine the column name for predictions
                if 'predicted_orders' in center_meal_forecast.columns:
                    pred_col = 'predicted_orders'
                elif 'prediction' in center_meal_forecast.columns:
                    pred_col = 'prediction'
                else:
                    pred_col = center_meal_forecast.columns[-1]  # Fallback to last column
                
                fig = px.line(
                    center_meal_forecast,
                    x='week',
                    y=pred_col,
                    title=f'Demand Forecast for Center {selected_center}, Meal {selected_meal}'
                )
                fig.update_layout(
                    xaxis_title='Week',
                    yaxis_title='Predicted Orders',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No forecast data available for the selected center and meal combination.")
        
    # Date-Specific Forecasting tab
    with date_tab:
        st.subheader("Date-Specific Forecasting")
        st.write("Forecast demand for a specific date, taking into account holidays and special events.")
        
        # Try to use interactive calendar
        try:
            import date_picker_component
            import holiday_data
            
            # Get date and location inputs using our interactive calendar
            selected_date, selected_country = date_picker_component.holiday_calendar_picker(
                title="Select Date for Forecasting",
                default_date=datetime.now().date()
            )
            
            # Get available centers and meals
            if st.session_state.data_loaded:
                train_data = st.session_state.train_data
                
                # Center and meal selection
                col1, col2 = st.columns(2)
                
                with col1:
                    center_ids = sorted(train_data['center_id'].unique())
                    selected_center = st.selectbox('Select Center ID', center_ids, key="date_center_select")
                
                with col2:
                    meal_ids = sorted(train_data['meal_id'].unique())
                    selected_meal = st.selectbox('Select Meal ID', meal_ids, key="date_meal_select")
                
                # Calculate demand factors
                country_mapping = {
                    "New York": "USA",
                    "London": "United Kingdom",
                    "Tokyo": "Japan",
                    "Mumbai": "India",
                    "Sydney": "Australia",
                    "Paris": "France",
                    "Beijing": "China",
                    "Berlin": "Germany",
                    "Mexico City": "Mexico",
                    "Cairo": "Egypt"
                }
                
                # Default city based on country
                reverse_mapping = {v: k for k, v in country_mapping.items()}
                default_city = reverse_mapping.get(selected_country, "New York")
                
                # Location selection
                selected_location = st.selectbox(
                    "Select Location",
                    ["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris", "Beijing", "Berlin", "Mexico City", "Cairo"],
                    index=["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris", "Beijing", "Berlin", "Mexico City", "Cairo"].index(default_city) 
                    if default_city in ["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris", "Beijing", "Berlin", "Mexico City", "Cairo"] else 0,
                    key="date_location_select"
                )
                
                # Calculate and display demand factor adjustments
                demand_factors = holiday_data.get_combined_factors(selected_date, selected_location, selected_country)
                
                st.subheader("Demand Adjustment Factors")
                
                # Display factors in columns
                factor_cols = st.columns(3)
                
                with factor_cols[0]:
                    holiday_name = demand_factors['holiday']
                    if holiday_name != 'No Holiday':
                        st.success(f"**Special Day:** {holiday_name}")
                        st.metric("Holiday Demand Impact", f"{int((demand_factors['holiday_factor']-1)*100)}%")
                    else:
                        st.write("**Special Day:** None")
                        st.metric("Holiday Demand Impact", "0%")
                
                with factor_cols[1]:
                    st.write("**Weather Conditions:**")
                    st.write(f"Temperature: {demand_factors['temperature']}°C")
                    st.write(f"Precipitation: {demand_factors['precipitation']} mm")
                    st.metric("Weather Demand Impact", f"{int((demand_factors['weather_factor']-1)*100)}%")
                
                with factor_cols[2]:
                    is_weekend = "Yes" if demand_factors['weekday_factor'] > 1 else "No"
                    st.write(f"**Weekend:** {is_weekend}")
                    st.metric("Combined Demand Impact", f"{int((demand_factors['combined_factor']-1)*100)}%")
                
                # Forecast for this specific date button
                if st.button("Generate Forecast for Selected Date"):
                    with st.spinner('Generating forecast for specific date...'):
                        # Simulating calculation with a progress bar
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.03)
                            progress_bar.progress(percent_complete + 1)
                        
                        try:
                            # Get baseline demand from existing predictions
                            if 'predictions' in st.session_state:
                                predictions = st.session_state.predictions
                                filtered_preds = predictions[
                                    (predictions['center_id'] == selected_center) &
                                    (predictions['meal_id'] == selected_meal)
                                ]
                                
                                if not filtered_preds.empty:
                                    # Determine the column name for predictions
                                    if 'predicted_orders' in filtered_preds.columns:
                                        pred_col = 'predicted_orders'
                                    elif 'prediction' in filtered_preds.columns:
                                        pred_col = 'prediction'
                                    else:
                                        pred_col = filtered_preds.columns[-1]
                                    
                                    # Get baseline demand
                                    baseline_demand = filtered_preds[pred_col].mean()
                                    
                                    # Adjust for demand factors
                                    combined_factor = demand_factors['combined_factor']
                                    adjusted_demand = baseline_demand * combined_factor
                                    
                                    # Display results
                                    st.subheader(f"Forecast Results for {selected_date.strftime('%A, %B %d, %Y')}")
                                    
                                    result_cols = st.columns(2)
                                    
                                    with result_cols[0]:
                                        st.metric("Baseline Demand", f"{baseline_demand:.2f} units")
                                        st.write(f"**Adjustment Factor:** {combined_factor:.2f}x")
                                    
                                    with result_cols[1]:
                                        st.metric(
                                            "Adjusted Demand", 
                                            f"{adjusted_demand:.2f} units",
                                            f"{(combined_factor-1)*100:.1f}%"
                                        )
                                    
                                    # Display factors breakdown
                                    st.subheader("Adjustment Factors Breakdown")
                                    
                                    factor_df = pd.DataFrame({
                                        'Factor': ['Holiday', 'Weather', 'Day of Week'],
                                        'Impact': [
                                            f"{(demand_factors['holiday_factor']-1)*100:.1f}%", 
                                            f"{(demand_factors['weather_factor']-1)*100:.1f}%",
                                            f"{(demand_factors['weekday_factor']-1)*100:.1f}%"
                                        ],
                                        'Value': [
                                            demand_factors['holiday_factor'],
                                            demand_factors['weather_factor'],
                                            demand_factors['weekday_factor']
                                        ]
                                    })
                                    
                                    st.dataframe(factor_df)
                                    
                                    # Show recommendations
                                    st.subheader("Recommendations")
                                    
                                    if combined_factor > 1.3:
                                        st.warning("High demand expected. Consider increasing inventory by at least 30%.")
                                    elif combined_factor > 1.1:
                                        st.info("Moderate increase in demand expected. Consider increasing inventory by 10-20%.")
                                    elif combined_factor < 0.9:
                                        st.info("Lower demand expected. Consider reducing inventory to avoid spoilage.")
                                    else:
                                        st.success("Demand is expected to be close to baseline. Standard inventory should be sufficient.")
                                    
                                    if holiday_name != 'No Holiday':
                                        st.write(f"**Note:** Due to {holiday_name}, special promotional offers may increase demand beyond the forecast.")
                                else:
                                    st.error("No prediction data available for the selected center and meal.")
                            else:
                                st.error("Please load predictions first before generating date-specific forecasts.")
                        except Exception as e:
                            st.error(f"Error generating date-specific forecast: {str(e)}")
            else:
                st.warning("Please load data first from the Data Loading & Exploration page.")
        except ImportError:
            st.error("Date-specific forecasting requires the date_picker_component and holiday_data modules.")

# Supply Chain Optimization page
elif page == 'Supply Chain Optimization':
    st.header('Supply Chain Optimization')
    
    if not st.session_state.predictions_generated or 'predictions' not in st.session_state:
        st.warning('Please generate forecast first from the Model Training & Forecasting page.')
    else:
        predictions = st.session_state.predictions
        
        st.subheader('Supply Chain Parameters')
        
        # Supply chain parameters
        col1, col2 = st.columns(2)
        
        with col1:
            safety_buffer = st.slider('Safety Buffer (%)', min_value=5, max_value=30, value=10, step=5) / 100
            lead_time = st.slider('Lead Time (days)', min_value=1, max_value=7, value=3, step=1)
            spoilage_rate = st.slider('Spoilage Rate (%)', min_value=1, max_value=15, value=5, step=1) / 100
            
        with col2:
            storage_cost = st.slider('Storage Cost (per unit per day)', min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            transport_cost = st.slider('Transport Cost (per unit)', min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            shortage_cost = st.slider('Shortage Cost (per unit)', min_value=5.0, max_value=20.0, value=10.0, step=1.0)
        
        # Optimize button
        if st.button('Run Supply Chain Optimization'):
            with st.spinner('Optimizing supply chain, please wait...'):
                # Simulating optimization with a progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(percent_complete + 1)
                
                try:
                    # Determine the column name for predictions
                    if 'predicted_orders' in predictions.columns:
                        pred_col = 'predicted_orders'
                    elif 'prediction' in predictions.columns:
                        pred_col = 'prediction'
                    else:
                        pred_col = predictions.columns[-1]  # Fallback to last column
                    
                    # Calculate inventory requirements with safety buffer
                    inventory_df = predictions.copy()
                    inventory_df['base_inventory'] = inventory_df[pred_col] * (1 + safety_buffer)
                    
                    # Account for lead time - simple approach for demo
                    inventory_df['lead_time_demand'] = inventory_df.groupby(['center_id', 'meal_id'])[pred_col].transform(
                        lambda x: x.rolling(min_periods=1, window=lead_time).mean()
                    )
                    
                    # Calculate required inventory
                    inventory_df['required_inventory'] = np.maximum(
                        inventory_df['base_inventory'],
                        inventory_df['lead_time_demand'] * lead_time * (1 + safety_buffer)
                    )
                    
                    # Account for spoilage
                    inventory_df['spoilage_adjustment'] = inventory_df['required_inventory'] * spoilage_rate
                    inventory_df['final_inventory'] = inventory_df['required_inventory'] + inventory_df['spoilage_adjustment']
                    
                    # Round to whole units
                    inventory_df['final_inventory'] = np.ceil(inventory_df['final_inventory']).astype(int)
                    
                    # Calculate costs
                    inventory_df['storage_cost'] = inventory_df['final_inventory'] * storage_cost * 7  # Weekly cost
                    inventory_df['transport_cost'] = inventory_df['final_inventory'] * transport_cost
                    
                    # Estimate potential shortages (simplified)
                    inventory_df['potential_shortage'] = np.maximum(0, inventory_df[pred_col] - inventory_df['final_inventory'])
                    inventory_df['shortage_cost'] = inventory_df['potential_shortage'] * shortage_cost
                    
                    # Total weekly cost
                    inventory_df['total_cost'] = inventory_df['storage_cost'] + inventory_df['transport_cost'] + inventory_df['shortage_cost']
                    
                    # Calculate service level (simplified)
                    total_demand = inventory_df[pred_col].sum()
                    total_potential_shortage = inventory_df['potential_shortage'].sum()
                    service_level = 1 - (total_potential_shortage / total_demand if total_demand > 0 else 0)
                    
                    # Store in session state
                    st.session_state.inventory_df = inventory_df
                    st.session_state.service_level = service_level
                    st.session_state.supply_chain_optimized = True
                    
                    st.success('Supply chain optimization completed successfully!')
                except Exception as e:
                    st.error(f'Error in supply chain optimization: {str(e)}')
        
        # Display optimization results
        if st.session_state.supply_chain_optimized and 'inventory_df' in st.session_state:
            st.subheader('Optimization Results')
            
            inventory_df = st.session_state.inventory_df
            service_level = st.session_state.service_level
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_inventory = inventory_df['final_inventory'].sum()
            total_cost = inventory_df['total_cost'].sum()
            avg_weekly_cost = total_cost / len(inventory_df['week'].unique())
            
            col1.metric("Total Required Inventory", f"{int(total_inventory):,}")
            col2.metric("Service Level", f"{service_level:.2%}")
            col3.metric("Total Cost", f"${total_cost:,.2f}")
            col4.metric("Avg Weekly Cost", f"${avg_weekly_cost:,.2f}")
            
            # Cost breakdown
            st.subheader('Cost Breakdown')
            
            cost_data = {
                'Cost Type': ['Storage', 'Transport', 'Shortage'],
                'Amount': [
                    inventory_df['storage_cost'].sum(),
                    inventory_df['transport_cost'].sum(),
                    inventory_df['shortage_cost'].sum()
                ]
            }
            cost_df = pd.DataFrame(cost_data)
            
            fig = px.pie(
                cost_df,
                values='Amount',
                names='Cost Type',
                title='Cost Distribution'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Inventory over time (for specific center/meal)
            st.subheader('Inventory Planning')
            
            # Filters for visualization
            col1, col2 = st.columns(2)
            
            with col1:
                center_ids = sorted(inventory_df['center_id'].unique())
                selected_center = st.selectbox('Select Center ID for Inventory Planning', center_ids)
            
            with col2:
                meal_ids = sorted(inventory_df['meal_id'].unique())
                selected_meal = st.selectbox('Select Meal ID for Inventory Planning', meal_ids)
            
            # Filter inventory data
            center_meal_inventory = inventory_df[
                (inventory_df['center_id'] == selected_center) &
                (inventory_df['meal_id'] == selected_meal)
            ]
            
            # Plot inventory plan
            if not center_meal_inventory.empty:
                # Determine the column name for predictions
                if 'predicted_orders' in center_meal_inventory.columns:
                    pred_col = 'predicted_orders'
                elif 'prediction' in center_meal_inventory.columns:
                    pred_col = 'prediction'
                else:
                    pred_col = center_meal_inventory.columns[-1]  # Fallback to last column
                
                # Prepare data for the plot
                plot_data = center_meal_inventory[['week', pred_col, 'final_inventory']].copy()
                plot_data = plot_data.rename(columns={
                    pred_col: 'Predicted Demand',
                    'final_inventory': 'Required Inventory'
                })
                
                # Melt the dataframe for plotly
                plot_data_melted = pd.melt(
                    plot_data,
                    id_vars=['week'],
                    value_vars=['Predicted Demand', 'Required Inventory'],
                    var_name='Type',
                    value_name='Units'
                )
                
                # Create the plot
                fig = px.line(
                    plot_data_melted,
                    x='week',
                    y='Units',
                    color='Type',
                    title=f'Inventory Plan for Center {selected_center}, Meal {selected_meal}'
                )
                
                fig.update_layout(
                    xaxis_title='Week',
                    yaxis_title='Units',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display tabular data
                st.write('Detailed Inventory Plan:')
                # Ensure no duplicate column names
                display_cols = ['week', pred_col, 'base_inventory', 
                               'lead_time_demand', 'final_inventory', 'total_cost']
                # Check for duplicates in the list
                unique_cols = []
                for col in display_cols:
                    if col not in unique_cols:
                        unique_cols.append(col)
                
                st.dataframe(center_meal_inventory[unique_cols].reset_index(drop=True))
            else:
                st.warning("No inventory data available for the selected center and meal combination.")

# Report Generation page
elif page == 'Report Generation':
    st.header('Comprehensive Report Generation')
    
    if not st.session_state.predictions_generated or 'predictions' not in st.session_state:
        st.warning('Please generate forecast first from the Model Training & Forecasting page.')
    else:
        st.subheader('Generate PDF Report')
        st.write("Generate a comprehensive PDF report with forecasts, visualizations, and optimization results.")
        
        # Import report generator
        import report_generator
        
        # Get data from session state
        predictions = st.session_state.predictions
        
        # Center and meal selection
        col1, col2 = st.columns(2)
        
        with col1:
            center_ids = sorted(predictions['center_id'].unique())
            selected_center = st.selectbox('Select Center ID for Report', center_ids, key="report_center")
        
        with col2:
            meal_ids = sorted(predictions['meal_id'].unique())
            selected_meal = st.selectbox('Select Meal ID for Report', meal_ids, key="report_meal")
        
        # Include optimization data
        include_optimization = st.checkbox('Include Supply Chain Optimization', 
                                         value=True if (hasattr(st.session_state, 'supply_chain_optimized') and 
                                                      st.session_state.supply_chain_optimized and 
                                                      'inventory_df' in st.session_state) else False)
        
        # Include date-specific forecast
        include_date_specific = st.checkbox('Include Date-Specific Forecast', value=False)
        
        if include_date_specific:
            # Date selection for forecast
            date_col1, date_col2 = st.columns(2)
            
            with date_col1:
                forecast_date = st.date_input(
                    "Select Date for Forecast",
                    value=datetime.now().date()
                )
            
            with date_col2:
                # Location input
                available_locations = ["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris", "Beijing", "Berlin", "Mexico City", "Cairo"]
                forecast_location = st.selectbox(
                    "Select Location",
                    options=available_locations,
                    key="report_location"
                )
                
                # Country selection
                country_mapping = {
                    "New York": "USA",
                    "London": "United Kingdom",
                    "Tokyo": "Japan",
                    "Mumbai": "India",
                    "Sydney": "Australia",
                    "Paris": "France",
                    "Beijing": "China",
                    "Berlin": "Germany",
                    "Mexico City": "Mexico",
                    "Cairo": "Egypt"
                }
                forecast_country = country_mapping.get(forecast_location, "global")
        
        # Generate report button
        if st.button('Generate PDF Report'):
            with st.spinner('Generating PDF report, please wait...'):
                try:
                    # Prepare date-specific forecast data if requested
                    date_specific_data = None
                    if include_date_specific:
                        try:
                            import holiday_data
                            
                            # Get demand factors
                            demand_factors = holiday_data.get_combined_factors(
                                forecast_date, 
                                forecast_location, 
                                forecast_country
                            )
                            
                            # Get baseline demand
                            filtered_preds = predictions[
                                (predictions['center_id'] == selected_center) &
                                (predictions['meal_id'] == selected_meal)
                            ]
                            
                            if not filtered_preds.empty:
                                # Determine the column name for predictions
                                if 'predicted_orders' in filtered_preds.columns:
                                    pred_col = 'predicted_orders'
                                elif 'prediction' in filtered_preds.columns:
                                    pred_col = 'prediction'
                                else:
                                    pred_col = filtered_preds.columns[-1]
                                
                                # Calculate demand
                                baseline_demand = filtered_preds[pred_col].mean()
                                combined_factor = demand_factors['combined_factor']
                                adjusted_demand = baseline_demand * combined_factor
                                
                                # Create date-specific data
                                date_specific_data = {
                                    'date_str': forecast_date.strftime('%A, %B %d, %Y'),
                                    'baseline_demand': baseline_demand,
                                    'adjusted_demand': adjusted_demand,
                                    'combined_factor': combined_factor,
                                    'demand_factors': demand_factors
                                }
                        except ImportError:
                            st.warning("Date-specific forecasting requires the holiday_data module.")
                            date_specific_data = None
                    
                    # Generate the report file
                    file_path = report_generator.generate_full_report(
                        predictions=predictions,
                        inventory_df=st.session_state.inventory_df if include_optimization and hasattr(st.session_state, 'inventory_df') else None,
                        service_level=st.session_state.service_level if include_optimization and hasattr(st.session_state, 'service_level') else None,
                        center_id=selected_center,
                        meal_id=selected_meal,
                        date_specific=date_specific_data
                    )
                    
                    # Provide download link
                    with open(file_path, "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.success('PDF report generated successfully!')
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=os.path.basename(file_path),
                        mime="application/pdf"
                    )
                    
                except Exception as e:
                    st.error(f'Error generating PDF report: {str(e)}')

# Climate Impact Analysis page
elif page == 'Climate Impact Analysis':
    st.header('Climate Impact Analysis')
    
    # Date and location for climate data
    st.subheader('Climate Data Parameters')
    
    # Try to import the interactive calendar component
    try:
        import date_picker_component
        
        # Create tabs for regular and interactive date selection
        date_tabs = st.tabs(["Regular Selection", "Interactive Calendar"])
        
        with date_tabs[0]:
            # Regular date selection
            col1, col2 = st.columns(2)
            
            with col1:
                # Date input for climate data
                analysis_date = st.date_input(
                    "Select Date for Climate Analysis",
                    value=datetime.now().date()
                )
            
            with col2:
                # Location input for climate data
                available_locations = ["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris", "Beijing", "Berlin", "Mexico City", "Cairo"]
                selected_location = st.selectbox(
                    "Select Location",
                    options=available_locations
                )
                
                # Country selection (for holiday data)
                country_mapping = {
                    "New York": "USA",
                    "London": "United Kingdom",
                    "Tokyo": "Japan",
                    "Mumbai": "India",
                    "Sydney": "Australia",
                    "Paris": "France",
                    "Beijing": "China",
                    "Berlin": "Germany",
                    "Mexico City": "Mexico",
                    "Cairo": "Egypt"
                }
                selected_country = country_mapping.get(selected_location, "global")
        
        with date_tabs[1]:
            # Interactive calendar with holiday recognition
            selected_date, cal_country = date_picker_component.holiday_calendar_picker(
                title="Select Date with Holiday Information",
                default_date=datetime.now().date()
            )
            
            # Country mapping for locations
            reverse_mapping = {v: k for k, v in country_mapping.items()}
            default_city = reverse_mapping.get(cal_country, "New York")
            
            # Location selection
            selected_location = st.selectbox(
                "Select Location",
                options=available_locations,
                index=available_locations.index(default_city) if default_city in available_locations else 0,
                key="location_calendar_view"
            )
            
            # Update variables for the rest of the page
            analysis_date = selected_date
            selected_country = country_mapping.get(selected_location, "global")
    
    except ImportError:
        # Fallback to regular date selection
        col1, col2 = st.columns(2)
        
        with col1:
            # Date input for climate data
            analysis_date = st.date_input(
                "Select Date for Climate Analysis",
                value=datetime.now().date()
            )
        
        with col2:
            # Location input for climate data
            available_locations = ["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris", "Beijing", "Berlin", "Mexico City", "Cairo"]
            selected_location = st.selectbox(
                "Select Location",
                options=available_locations
            )
            
            # Country selection (for holiday data)
            country_mapping = {
                "New York": "USA",
                "London": "United Kingdom",
                "Tokyo": "Japan",
                "Mumbai": "India",
                "Sydney": "Australia",
                "Paris": "France",
                "Beijing": "China",
                "Berlin": "Germany",
                "Mexico City": "Mexico",
                "Cairo": "Egypt"
            }
            selected_country = country_mapping.get(selected_location, "global")
    
    # Get climate and holiday data
    try:
        import holiday_data
        demand_factors = holiday_data.get_combined_factors(analysis_date, selected_location, selected_country)
        
        # Display selected parameters and results
        st.info(f"Climate analysis for {selected_location} on {analysis_date.strftime('%B %d, %Y')}")
        
        # Display holiday and climate factors
        factor_cols = st.columns(3)
        
        with factor_cols[0]:
            holiday_name = demand_factors['holiday']
            if holiday_name != 'No Holiday':
                st.success(f"**Special Day:** {holiday_name}")
                st.metric("Holiday Demand Impact", f"{int((demand_factors['holiday_factor']-1)*100)}%")
            else:
                st.write("**Special Day:** None")
                st.metric("Holiday Demand Impact", "0%")
        
        with factor_cols[1]:
            st.write("**Weather Conditions:**")
            st.write(f"Temperature: {demand_factors['temperature']}°C")
            st.write(f"Precipitation: {demand_factors['precipitation']} mm")
            st.metric("Weather Demand Impact", f"{int((demand_factors['weather_factor']-1)*100)}%")
        
        with factor_cols[2]:
            is_weekend = "Yes" if demand_factors['weekday_factor'] > 1 else "No"
            st.write(f"**Weekend:** {is_weekend}")
            st.metric("Combined Demand Impact", f"{int((demand_factors['combined_factor']-1)*100)}%")
        
        # Add horizontal line
        st.markdown("---")
    except ImportError:
        st.warning("Holiday data module not available. Unable to show climate impact on demand.")
    
    # Continue with supply chain optimization check
    if not 'supply_chain_optimized' in st.session_state or not st.session_state.supply_chain_optimized:
        st.warning('Please run supply chain optimization first from the Supply Chain Optimization page.')
    else:
        inventory_df = st.session_state.inventory_df
        
        st.subheader('Environmental Parameters')
        
        # Environmental parameters
        col1, col2 = st.columns(2)
        
        with col1:
            transport_method = st.selectbox(
                'Transport Method',
                ['Truck', 'Rail', 'Ship', 'Air'],
                index=0
            )
            transport_distance = st.slider('Average Transport Distance (km)', min_value=50, max_value=500, value=250, step=50)
            
        with col2:
            renewable_energy = st.slider('Renewable Energy Use (%)', min_value=0, max_value=100, value=30, step=10) / 100
            packaging_type = st.selectbox(
                'Packaging Type',
                ['Plastic', 'Paper', 'Biodegradable', 'Reusable'],
                index=0
            )
        
        # Emission factors (kg CO2e)
        emission_factors = {
            'transport': {
                'Truck': 0.1,
                'Rail': 0.03,
                'Ship': 0.02,
                'Air': 2.0
            },
            'storage': {
                'standard': 0.5,
                'renewable': 0.2
            },
            'packaging': {
                'Plastic': 0.5,
                'Paper': 0.25,
                'Biodegradable': 0.15,
                'Reusable': 0.1
            },
            'food_waste': 2.5
        }
        
        # Calculate climate impact button
        if st.button('Calculate Climate Impact'):
            with st.spinner('Calculating climate impact, please wait...'):
                # Simulating calculation with a progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                
                try:
                    # Determine the column name for predictions
                    if 'predicted_orders' in inventory_df.columns:
                        pred_col = 'predicted_orders'
                    elif 'prediction' in inventory_df.columns:
                        pred_col = 'prediction'
                    else:
                        pred_col = inventory_df.columns[-1]  # Fallback to last column
                    
                    # Get total units and waste
                    total_units = inventory_df['final_inventory'].sum()
                    total_demand = inventory_df[pred_col].sum()
                    total_waste = total_units * inventory_df['spoilage_adjustment'].sum() / inventory_df['final_inventory'].sum() if total_units > 0 else 0
                    
                    # Calculate waste rate
                    waste_rate = total_waste / total_units if total_units > 0 else 0
                    
                    # Calculate emissions
                    # Transport emissions
                    transport_factor = emission_factors['transport'].get(transport_method, 0.1)
                    transport_emissions = total_units * transport_distance * transport_factor / 1000  # Convert to tonnes
                    
                    # Storage emissions
                    standard_factor = emission_factors['storage']['standard']
                    renewable_factor = emission_factors['storage']['renewable']
                    weighted_factor = (1 - renewable_energy) * standard_factor + renewable_energy * renewable_factor
                    
                    # Simplification: assume avg storage of 3.5 days (half a week) for each unit
                    storage_emissions = total_units * 3.5 * weighted_factor / 1000  # Convert to tonnes
                    
                    # Packaging emissions
                    packaging_factor = emission_factors['packaging'].get(packaging_type, 0.5)
                    packaging_emissions = total_units * packaging_factor / 1000  # Convert to tonnes
                    
                    # Waste emissions
                    waste_factor = emission_factors['food_waste']
                    waste_emissions = total_waste * waste_factor / 1000  # Convert to tonnes
                    
                    # Total emissions
                    total_emissions = transport_emissions + storage_emissions + packaging_emissions + waste_emissions
                    
                    # Emissions per unit
                    emissions_per_unit = total_emissions * 1000 / total_demand if total_demand > 0 else 0
                    
                    # Carbon intensity (emissions per $ revenue)
                    # Assuming average revenue of $10 per unit
                    avg_revenue_per_unit = 10
                    total_revenue = total_demand * avg_revenue_per_unit
                    carbon_intensity = total_emissions * 1000 / total_revenue if total_revenue > 0 else 0
                    
                    # Calculate sustainability score (0-100)
                    # Carbon intensity score (0-30)
                    carbon_score = max(0, 30 * (1 - carbon_intensity / 1.5))
                    
                    # Waste rate score (0-25)
                    waste_score = max(0, 25 * (1 - waste_rate / 0.3))
                    
                    # Renewable energy score (0-20)
                    energy_score = 20 * renewable_energy
                    
                    # Packaging score (0-15)
                    packaging_scores = {
                        'Plastic': 3,
                        'Paper': 8,
                        'Biodegradable': 12,
                        'Reusable': 15
                    }
                    packaging_score = packaging_scores.get(packaging_type, 3)
                    
                    # Transport method score (0-10)
                    transport_scores = {
                        'Air': 1,
                        'Truck': 5,
                        'Ship': 7,
                        'Rail': 10
                    }
                    transport_score = transport_scores.get(transport_method, 5)
                    
                    # Total sustainability score
                    sustainability_score = carbon_score + waste_score + energy_score + packaging_score + transport_score
                    
                    # Store results in session state
                    st.session_state.climate_impact = {
                        'total_emissions': total_emissions,
                        'emissions_per_unit': emissions_per_unit,
                        'carbon_intensity': carbon_intensity,
                        'emissions_breakdown': {
                            'Transport': transport_emissions,
                            'Storage': storage_emissions,
                            'Packaging': packaging_emissions,
                            'Food Waste': waste_emissions
                        },
                        'sustainability_score': sustainability_score
                    }
                    
                    st.session_state.climate_impact_calculated = True
                    
                    st.success('Climate impact analysis completed successfully!')
                except Exception as e:
                    st.error(f'Error in climate impact analysis: {str(e)}')
        
        # Display climate impact results
        if 'climate_impact_calculated' in st.session_state and st.session_state.climate_impact_calculated:
            st.subheader('Climate Impact Results')
            
            climate_impact = st.session_state.climate_impact
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Total Emissions", f"{climate_impact['total_emissions']:.2f} tonnes CO2e")
            col2.metric("Emissions per Unit", f"{climate_impact['emissions_per_unit']:.2f} kg CO2e")
            col3.metric("Carbon Intensity", f"{climate_impact['carbon_intensity']:.2f} kg CO2e/$")
            
            # Sustainability score gauge
            st.subheader('Sustainability Score')
            
            score = climate_impact['sustainability_score']
            
            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sustainability Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Emissions breakdown
            st.subheader('Emissions Breakdown')
            
            emissions_data = {
                'Source': list(climate_impact['emissions_breakdown'].keys()),
                'Emissions (tonnes CO2e)': list(climate_impact['emissions_breakdown'].values())
            }
            emissions_df = pd.DataFrame(emissions_data)
            
            # Create bar chart
            fig = px.bar(
                emissions_df,
                x='Source',
                y='Emissions (tonnes CO2e)',
                title='Emissions by Source',
                color='Source'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate recommendations
            st.subheader('Sustainability Recommendations')
            
            # Find highest emission source
            highest_source = max(climate_impact['emissions_breakdown'].items(), key=lambda x: x[1])
            
            recommendations = []
            
            # Recommendations based on highest emission source
            if highest_source[0] == 'Transport':
                if transport_method == 'Air':
                    recommendations.append(
                        "Switch from air freight to rail or ship transport to significantly reduce emissions."
                    )
                elif transport_method == 'Truck':
                    recommendations.append(
                        "Consider using rail transport for long distances or optimizing delivery routes."
                    )
                    
            elif highest_source[0] == 'Storage':
                if renewable_energy < 0.5:
                    recommendations.append(
                        f"Increase renewable energy use from {renewable_energy*100:.0f}% to at least 50% in storage facilities."
                    )
                recommendations.append(
                    "Improve warehouse energy efficiency with better insulation and cold storage technologies."
                )
                    
            elif highest_source[0] == 'Packaging':
                if packaging_type == 'Plastic':
                    recommendations.append(
                        "Switch from plastic to biodegradable or reusable packaging to reduce emissions."
                    )
                elif packaging_type == 'Paper':
                    recommendations.append(
                        "Consider reusable packaging options to further reduce packaging emissions."
                    )
                    
            elif highest_source[0] == 'Food Waste':
                recommendations.append(
                    f"Current waste rate is contributing significantly to emissions. " +
                    "Improve inventory management and implement better forecasting."
                )
            
            # General recommendations
            if climate_impact['carbon_intensity'] > 0.8:
                recommendations.append(
                    f"Carbon intensity ({climate_impact['carbon_intensity']:.2f} kg CO2e/$) is above industry average. " +
                    "Consider setting emission reduction targets."
                )
            
            if renewable_energy < 0.3:
                recommendations.append(
                    "Increase use of renewable energy across the supply chain."
                )
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

# Email Notifications page
elif page == 'Email Notifications':
    st.header('Email Notifications')
    
    # Import email validation (with try/except)
    try:
        from email_validator import validate_email, EmailNotValidError
    except ImportError:
        # Define placeholder functions if the module isn't available
        st.error("Email validation module not installed. Some features might not work correctly.")
        
        def validate_email(email):
            """Simple fallback email validator"""
            class ValidationResult:
                def __init__(self, email):
                    self.email = email
            if '@' in email and '.' in email:
                return ValidationResult(email)
            raise Exception("Invalid email format")
        
        class EmailNotValidError(Exception):
            pass
    
    # Import database functions
    try:
        import database
    except ImportError:
        st.error("Database module not available. Email notifications will not work correctly.")
        database = None
    
    # Check if SMTP is configured
    smtp_configured = False
    if 'SMTP_USERNAME' in os.environ and 'SMTP_PASSWORD' in os.environ:
        smtp_configured = True
    
    if not smtp_configured:
        st.warning("Email notifications require SMTP configuration. Please ask the administrator to configure SMTP settings.")
        
        # For demonstration purposes, allow configuring SMTP in the app
        with st.expander("Configure SMTP Settings (For demonstration only)"):
            st.write("In a production environment, these settings would be configured securely using environment variables.")
            
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
            smtp_username = st.text_input("SMTP Username (Email)", placeholder="your-email@gmail.com")
            smtp_password = st.text_input("SMTP Password", type="password")
            
            if st.button("Save SMTP Settings"):
                if smtp_username and smtp_password:
                    os.environ['SMTP_SERVER'] = smtp_server
                    os.environ['SMTP_PORT'] = str(smtp_port)
                    os.environ['SMTP_USERNAME'] = smtp_username
                    os.environ['SMTP_PASSWORD'] = smtp_password
                    os.environ['SENDER_EMAIL'] = smtp_username
                    st.success("SMTP settings saved successfully!")
                    smtp_configured = True
                else:
                    st.error("Please provide both username and password.")
    
    # User registration section
    st.subheader("User Registration")
    
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input("Your Name", placeholder="John Doe")
    with col2:
        user_email = st.text_input("Your Email", placeholder="your-email@example.com")
    
    register_button = st.button("Register for Notifications")
    
    if register_button:
        if not user_name or not user_email:
            st.error("Please provide both name and email.")
        else:
            # Validate email
            try:
                validated_email = validate_email(user_email).email
                
                # Register user
                user = database.create_user(user_name, validated_email)
                
                if user:
                    st.session_state.current_user = user
                    st.success(f"Registration successful! You can now set up notifications.")
                else:
                    st.error("Failed to register user. Please try again.")
                    
            except EmailNotValidError as e:
                st.error(f"Invalid email: {str(e)}")
    
    # Check if user is logged in
    current_user = st.session_state.get('current_user', None)
    
    if current_user:
        st.write(f"Logged in as: **{current_user['name']}** ({current_user['email']})")
        
        # Notification setup section
        st.subheader("Set Up Notifications")
        
        # Need predictions to set up notifications
        if not 'predictions' in st.session_state or st.session_state.predictions is None:
            st.warning("Please load predictions from the 'Model Training & Forecasting' page first.")
        else:
            predictions = st.session_state.predictions
            
            # Set up form for notification preferences
            with st.form("notification_form"):
                st.write("Create a new notification alert")
                
                # Determine prediction column
                if 'predicted_orders' in predictions.columns:
                    pred_col = 'predicted_orders'
                elif 'prediction' in predictions.columns:
                    pred_col = 'prediction'
                elif 'predicted_demand' in predictions.columns:
                    pred_col = 'predicted_demand'
                else:
                    pred_col = predictions.columns[-1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    center_ids = sorted(predictions['center_id'].unique())
                    selected_center = st.selectbox('Center ID', center_ids, key='notif_center')
                
                with col2:
                    meal_ids = sorted(predictions['meal_id'].unique())
                    selected_meal = st.selectbox('Meal ID', meal_ids, key='notif_meal')
                
                # Filter predictions for selected center/meal
                filtered_preds = predictions[
                    (predictions['center_id'] == selected_center) &
                    (predictions['meal_id'] == selected_meal)
                ]
                
                # Show current predictions
                if not filtered_preds.empty:
                    latest_pred = filtered_preds.iloc[-1][pred_col]
                    st.info(f"Current predicted demand: {latest_pred:.2f} units")
                
                # Threshold type and value
                col1, col2 = st.columns(2)
                
                with col1:
                    threshold_type = st.selectbox(
                        'Alert Type',
                        ['above', 'below', 'change_rate'],
                        format_func=lambda x: {
                            'above': 'Demand Above Threshold',
                            'below': 'Demand Below Threshold',
                            'change_rate': 'Change Rate Exceeds Threshold'
                        }.get(x, x)
                    )
                
                with col2:
                    default_value = 100.0
                    if not filtered_preds.empty and 'latest_pred' in locals():
                        default_value = float(latest_pred)
                    
                    if threshold_type in ['above', 'below']:
                        threshold_value = st.number_input('Threshold Value (units)', min_value=0.0, value=default_value, step=10.0)
                    else:
                        threshold_value = st.number_input('Threshold Value (%)', min_value=0.0, value=10.0, step=1.0)
                
                # Submit button
                submit_button = st.form_submit_button("Save Notification Setting")
                
                if submit_button:
                    # Save notification preference
                    pref = database.add_notification_preference(
                        current_user['id'],
                        selected_center,
                        selected_meal,
                        threshold_type,
                        threshold_value
                    )
                    
                    if pref:
                        st.success("Notification setting saved successfully!")
                    else:
                        st.error("Failed to save notification setting. Please try again.")
            
            # Display existing notification preferences
            st.subheader("Your Notification Settings")
            
            preferences = database.get_user_preferences(current_user['id'])
            
            if preferences:
                for i, pref in enumerate(preferences):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            if pref['threshold_type'] == 'above':
                                condition = f"Demand > {pref['threshold_value']:.2f} units"
                            elif pref['threshold_type'] == 'below':
                                condition = f"Demand < {pref['threshold_value']:.2f} units"
                            else:
                                condition = f"Change rate > {pref['threshold_value']:.2f}%"
                                
                            st.write(f"Center {pref['center_id']}, Meal {pref['meal_id']}: {condition}")
                        
                        with col2:
                            created_at = pref['created_at'].strftime('%Y-%m-%d') if hasattr(pref['created_at'], 'strftime') else pref['created_at']
                            st.write(f"Added: {created_at}")
                        
                        with col3:
                            if st.button("Delete", key=f"del_{pref['id']}"):
                                if database.delete_preference(pref['id'], current_user['id']):
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to delete preference")
                
                # Check notifications against current predictions
                if st.button("Check Notifications Now"):
                    notifications = database.check_thresholds(predictions)
                    if notifications:
                        st.success(f"Found {len(notifications)} notifications that would be sent!")
                        
                        for notif in notifications:
                            st.info(notif['message'])
                            
                        # Option to send test emails
                        if smtp_configured and st.button("Send Test Emails"):
                            from email_service import process_pending_notifications
                            sent_count = process_pending_notifications(notifications)
                            if sent_count > 0:
                                st.success(f"Successfully sent {sent_count} test emails!")
                            else:
                                st.error("Failed to send test emails. Please check SMTP configuration.")
                    else:
                        st.info("No notification thresholds were triggered.")
            else:
                st.write("You don't have any notification settings yet.")
    else:
        st.info("Please register to set up email notifications.")

if __name__ == "__main__":
    pass