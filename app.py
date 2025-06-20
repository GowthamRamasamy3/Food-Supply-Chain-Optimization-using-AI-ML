import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime

from data_processor import DataProcessor
from model import HybridModel
from supply_chain_optimizer import SupplyChainOptimizer
from climate_impact import ClimateImpactAnalyzer
from evaluation import ModelEvaluator
from utils import load_data, save_results, get_forecast_weeks

# Set page configuration
st.set_page_config(
    page_title="Food Demand Forecasting & Supply Chain Optimization",
    page_icon="🍲",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_generated' not in st.session_state:
    st.session_state.predictions_generated = False
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = None
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None

# Main app header
st.title('🍲 Food Demand Forecasting & Supply Chain Optimization')
st.write('Predict meal demand and optimize supply chain operations using hybrid ML models')

# Sidebar for navigation and controls
st.sidebar.title('Navigation')
page = st.sidebar.radio(
    'Select a page:',
    ['Data Loading & Exploration', 'Model Training & Forecasting', 'Supply Chain Optimization', 'Climate Impact Analysis', 'About']
)

# Data Loading & Exploration page
if page == 'Data Loading & Exploration':
    st.header('Data Loading & Exploration')
    
    # File uploader for historical data
    st.subheader('Upload Historical Data')
    uploaded_file = st.file_uploader("Upload historical meal demand data (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load and process the data
            with st.spinner('Loading and processing data...'):
                data = load_data(uploaded_file)
                
                # Initialize data processor
                data_processor = DataProcessor()
                processed_data = data_processor.preprocess_data(data)
                
                # Store in session state
                st.session_state.data = data
                st.session_state.processed_data = processed_data
                st.session_state.data_processor = data_processor
                st.session_state.data_loaded = True
                
                st.success('Data loaded and processed successfully!')
                
                # Display basic statistics
                st.subheader('Dataset Overview')
                st.write(f"Data shape: {data.shape}")
                st.write(f"Time period: Weeks 1 to {data['week'].max()}")
                st.write(f"Number of meal centers: {data['center_id'].nunique()}")
                st.write(f"Number of meal types: {data['meal_id'].nunique()}")
                
                # Data exploration tabs
                tab1, tab2, tab3, tab4 = st.tabs(['Data Sample', 'Feature Statistics', 'Meal Demand Patterns', 'Correlation Analysis'])
                
                with tab1:
                    st.dataframe(data.head(10))
                
                with tab2:
                    st.subheader('Statistical Summary')
                    st.dataframe(data.describe())
                    
                    # Missing values
                    missing_values = data.isnull().sum()
                    if missing_values.sum() > 0:
                        st.subheader('Missing Values')
                        st.dataframe(missing_values[missing_values > 0])
                
                with tab3:
                    st.subheader('Meal Demand Patterns')
                    
                    # Time series visualization
                    center_ids = sorted(data['center_id'].unique())
                    meal_ids = sorted(data['meal_id'].unique())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_center = st.selectbox('Select Center ID', center_ids)
                    with col2:
                        selected_meal = st.selectbox('Select Meal ID', meal_ids)
                    
                    filtered_data = data[(data['center_id'] == selected_center) & 
                                         (data['meal_id'] == selected_meal)]
                    
                    if not filtered_data.empty:
                        fig = px.line(
                            filtered_data, 
                            x='week', 
                            y='num_orders',
                            title=f'Demand Pattern for Center {selected_center}, Meal {selected_meal}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Seasonality/periodicity check
                        if len(filtered_data) > 10:
                            st.subheader('Seasonality Check')
                            plt.figure(figsize=(10, 6))
                            pd.plotting.autocorrelation_plot(filtered_data['num_orders'])
                            st.pyplot(plt.gcf())
                    else:
                        st.warning("No data available for the selected center and meal combination.")
                
                with tab4:
                    st.subheader('Correlation Analysis')
                    
                    # Calculate correlation
                    numeric_data = data.select_dtypes(include=['number'])
                    corr = numeric_data.corr()
                    
                    # Plot heatmap
                    fig = px.imshow(
                        corr, 
                        color_continuous_scale='RdBu_r',
                        labels=dict(color="Correlation"),
                        title='Feature Correlation Matrix'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Most correlated features with demand
                    if 'num_orders' in numeric_data.columns:
                        st.subheader('Features most correlated with demand (num_orders)')
                        demand_corr = corr['num_orders'].sort_values(ascending=False)
                        st.dataframe(demand_corr)
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file containing historical meal demand data.")

# Model Training & Forecasting page
elif page == 'Model Training & Forecasting':
    st.header('Model Training & Forecasting')
    
    if not st.session_state.data_loaded:
        st.warning("Please load and process data first.")
    else:
        # Model configuration
        st.subheader('Model Configuration')
        
        col1, col2 = st.columns(2)
        with col1:
            lstm_units = st.slider('LSTM Units', min_value=32, max_value=256, value=128, step=32)
            lstm_dropout = st.slider('LSTM Dropout Rate', min_value=0.1, max_value=0.5, value=0.2, step=0.1)
            sequence_length = st.slider('Sequence Length (weeks)', min_value=4, max_value=24, value=12, step=4)
            
        with col2:
            xgb_max_depth = st.slider('XGBoost Max Depth', min_value=3, max_value=10, value=6, step=1)
            xgb_learning_rate = st.slider('XGBoost Learning Rate', min_value=0.01, max_value=0.3, value=0.1, step=0.05)
            batch_size = st.slider('Batch Size', min_value=16, max_value=128, value=32, step=16)
        
        epochs = st.slider('Training Epochs', min_value=10, max_value=100, value=50, step=10)
        
        # Train-test split configuration
        st.subheader('Train-Test Split')
        test_size = st.slider('Test Set Size (%)', min_value=10, max_value=30, value=20, step=5) / 100
        
        # Train model button
        if st.button('Train Hybrid Model'):
            try:
                with st.spinner('Training model... This may take several minutes.'):
                    # Initialize the hybrid model
                    hybrid_model = HybridModel(
                        lstm_units=lstm_units,
                        lstm_dropout=lstm_dropout,
                        sequence_length=sequence_length,
                        xgb_max_depth=xgb_max_depth,
                        xgb_learning_rate=xgb_learning_rate
                    )
                    
                    # Train the model
                    data_processor = st.session_state.data_processor
                    X_train, X_test, y_train, y_test, feature_names = data_processor.prepare_training_data(
                        st.session_state.processed_data, 
                        sequence_length=sequence_length,
                        test_size=test_size
                    )
                    
                    hybrid_model.fit(
                        X_train, 
                        y_train, 
                        X_test, 
                        y_test,
                        batch_size=batch_size,
                        epochs=epochs,
                        feature_names=feature_names
                    )
                    
                    # Store model in session state
                    st.session_state.hybrid_model = hybrid_model
                    st.session_state.model_trained = True
                    
                    # Evaluate the model
                    evaluator = ModelEvaluator()
                    y_pred = hybrid_model.predict(X_test)
                    metrics = evaluator.calculate_metrics(y_test, y_pred)
                    
                    # Store evaluation results
                    st.session_state.evaluation_metrics = metrics
                    
                    st.success('Model training completed!')
                
                # Display evaluation metrics
                st.subheader('Model Evaluation Metrics')
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': list(metrics.values())
                })
                st.dataframe(metrics_df)
                
                # Plot actual vs predicted
                st.subheader('Actual vs Predicted (Test Set)')
                fig = px.scatter(
                    x=y_test.flatten(), 
                    y=y_pred.flatten(),
                    labels={'x': 'Actual Demand', 'y': 'Predicted Demand'},
                    title='Actual vs Predicted Demand'
                )
                fig.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=min(y_test.flatten()), y0=min(y_test.flatten()),
                    x1=max(y_test.flatten()), y1=max(y_test.flatten())
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
        
        # Forecasting section
        st.header('Demand Forecasting')
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first.")
        else:
            # Forecast configuration
            st.subheader('Forecast Configuration')
            forecast_horizon = st.slider('Forecast Horizon (weeks)', min_value=1, max_value=12, value=4, step=1)
            
            # Generate forecast button
            if st.button('Generate Forecast'):
                try:
                    with st.spinner('Generating forecast...'):
                        # Get the latest data
                        data = st.session_state.processed_data
                        hybrid_model = st.session_state.hybrid_model
                        data_processor = st.session_state.data_processor
                        
                        # Generate forecast for each center and meal combination
                        forecast_results = []
                        
                        # Get unique center_id and meal_id combinations
                        center_meal_combos = data[['center_id', 'meal_id']].drop_duplicates()
                        
                        # Determine the next weeks to forecast
                        last_week = data['week'].max()
                        forecast_weeks = get_forecast_weeks(last_week, forecast_horizon)
                        
                        # Generate forecast for each center-meal combination
                        for _, row in center_meal_combos.iterrows():
                            center_id = row['center_id']
                            meal_id = row['meal_id']
                            
                            # Prepare the forecast input
                            forecast_input = data_processor.prepare_forecast_input(
                                data, 
                                center_id, 
                                meal_id, 
                                sequence_length=sequence_length
                            )
                            
                            if forecast_input is not None:
                                # Generate predictions for the forecast horizon
                                predictions = hybrid_model.forecast(
                                    forecast_input, 
                                    forecast_horizon=forecast_horizon
                                )
                                
                                # Create forecast records
                                for i, week in enumerate(forecast_weeks):
                                    forecast_results.append({
                                        'week': week,
                                        'center_id': center_id,
                                        'meal_id': meal_id,
                                        'forecasted_demand': max(0, round(predictions[i])) # Ensure demand is non-negative and rounded
                                    })
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame(forecast_results)
                        
                        # Store forecast results
                        st.session_state.forecast_df = forecast_df
                        st.session_state.predictions_generated = True
                        
                        st.success('Forecast generated successfully!')
                        
                        # Display forecast results
                        st.subheader('Demand Forecast Results')
                        st.dataframe(forecast_df)
                        
                        # Download forecast as CSV
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast CSV",
                            data=csv,
                            file_name=f"meal_demand_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        st.subheader('Forecast Visualization')
                        
                        # Filter options
                        center_ids = sorted(forecast_df['center_id'].unique())
                        meal_ids = sorted(forecast_df['meal_id'].unique())
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            viz_center = st.selectbox('Select Center ID for Visualization', center_ids, key='viz_center')
                        with col2:
                            viz_meal = st.selectbox('Select Meal ID for Visualization', meal_ids, key='viz_meal')
                        
                        # Filter forecast data
                        filtered_forecast = forecast_df[
                            (forecast_df['center_id'] == viz_center) & 
                            (forecast_df['meal_id'] == viz_meal)
                        ]
                        
                        # Get historical data for the same center-meal
                        historical_data = st.session_state.data[
                            (st.session_state.data['center_id'] == viz_center) & 
                            (st.session_state.data['meal_id'] == viz_meal)
                        ][['week', 'num_orders']]
                        
                        # Create a combined visualization of historical and forecasted demand
                        if not historical_data.empty and not filtered_forecast.empty:
                            fig = go.Figure()
                            
                            # Add historical data
                            fig.add_trace(go.Scatter(
                                x=historical_data['week'],
                                y=historical_data['num_orders'],
                                mode='lines+markers',
                                name='Historical Demand',
                                line=dict(color='blue')
                            ))
                            
                            # Add forecast data
                            fig.add_trace(go.Scatter(
                                x=filtered_forecast['week'],
                                y=filtered_forecast['forecasted_demand'],
                                mode='lines+markers',
                                name='Forecasted Demand',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title=f'Historical and Forecasted Demand for Center {viz_center}, Meal {viz_meal}',
                                xaxis_title='Week',
                                yaxis_title='Demand (Number of Orders)',
                                legend=dict(x=0, y=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No data available for the selected center and meal combination.")
                
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

# Supply Chain Optimization page
elif page == 'Supply Chain Optimization':
    st.header('Supply Chain Optimization')
    
    if not st.session_state.predictions_generated:
        st.warning("Please generate demand forecasts first.")
    else:
        # Display demand forecast summary
        st.subheader('Demand Forecast Summary')
        forecast_df = st.session_state.forecast_df
        
        # Aggregate forecast by week
        weekly_forecast = forecast_df.groupby('week')['forecasted_demand'].sum().reset_index()
        
        # Display weekly forecast
        fig = px.bar(
            weekly_forecast,
            x='week',
            y='forecasted_demand',
            title='Total Weekly Forecasted Demand',
            labels={'forecasted_demand': 'Total Demand', 'week': 'Week'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Supply Chain Optimization Configuration
        st.subheader('Supply Chain Configuration')
        
        col1, col2 = st.columns(2)
        with col1:
            safety_buffer = st.slider('Safety Buffer (%)', min_value=5, max_value=30, value=10, step=5)
            lead_time = st.slider('Lead Time (days)', min_value=1, max_value=14, value=3, step=1)
        
        with col2:
            spoilage_rate = st.slider('Average Spoilage Rate (%)', min_value=1, max_value=15, value=5, step=1)
            transport_efficiency = st.slider('Transportation Efficiency (%)', min_value=70, max_value=100, value=90, step=5)
        
        # Cost factors
        st.subheader('Cost Factors')
        col1, col2, col3 = st.columns(3)
        with col1:
            storage_cost = st.number_input('Storage Cost per Unit per Day ($)', min_value=0.01, max_value=10.0, value=0.50, step=0.05)
        with col2:
            transport_cost = st.number_input('Transport Cost per Unit ($)', min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        with col3:
            shortage_cost = st.number_input('Shortage Cost per Unit ($)', min_value=1.0, max_value=50.0, value=10.0, step=1.0)
        
        # Optimize button
        if st.button('Run Supply Chain Optimization'):
            try:
                with st.spinner('Optimizing supply chain...'):
                    # Initialize the optimizer
                    optimizer = SupplyChainOptimizer(
                        safety_buffer=safety_buffer/100,
                        lead_time=lead_time,
                        spoilage_rate=spoilage_rate/100,
                        transport_efficiency=transport_efficiency/100,
                        storage_cost=storage_cost,
                        transport_cost=transport_cost,
                        shortage_cost=shortage_cost
                    )
                    
                    # Run optimization
                    optimization_results = optimizer.optimize(forecast_df)
                    
                    # Store results
                    st.session_state.optimization_results = optimization_results
                    st.session_state.optimization_done = True
                    
                    st.success('Supply chain optimization completed!')
                
                # Display optimization results
                st.subheader('Inventory Requirements')
                st.dataframe(optimization_results['inventory_requirements'])
                
                # Inventory timeline visualization
                st.subheader('Inventory Timeline')
                fig = px.line(
                    optimization_results['inventory_timeline'],
                    title='Projected Inventory Levels Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost breakdown
                st.subheader('Cost Breakdown')
                cost_df = pd.DataFrame({
                    'Cost Type': list(optimization_results['cost_breakdown'].keys()),
                    'Amount ($)': list(optimization_results['cost_breakdown'].values())
                })
                
                fig = px.pie(
                    cost_df,
                    values='Amount ($)',
                    names='Cost Type',
                    title='Supply Chain Cost Breakdown'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download optimization results
                csv = optimization_results['inventory_requirements'].to_csv(index=False)
                st.download_button(
                    label="Download Inventory Requirements CSV",
                    data=csv,
                    file_name=f"inventory_requirements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Optimization insights
                st.subheader('Optimization Insights')
                st.write(f"**Total Supply Chain Cost:** ${optimization_results['total_cost']:,.2f}")
                st.write(f"**Average Inventory Level:** {optimization_results['avg_inventory']:.1f} units")
                st.write(f"**Service Level:** {optimization_results['service_level']*100:.1f}%")
                st.write(f"**Total Waste:** {optimization_results['total_waste']:.1f} units (due to spoilage)")
                
                # Recommendations
                st.subheader('Recommendations')
                for recommendation in optimization_results['recommendations']:
                    st.info(recommendation)
            
            except Exception as e:
                st.error(f"Error optimizing supply chain: {str(e)}")

# Climate Impact Analysis page
elif page == 'Climate Impact Analysis':
    st.header('Climate Impact Analysis')
    
    if not st.session_state.optimization_done:
        st.warning("Please complete supply chain optimization first.")
    else:
        # Initialize climate impact analyzer
        climate_analyzer = ClimateImpactAnalyzer()
        
        # Get optimization results
        optimization_results = st.session_state.optimization_results
        forecast_df = st.session_state.forecast_df
        
        # Configuration options
        st.subheader('Climate Impact Configuration')
        
        col1, col2 = st.columns(2)
        with col1:
            transport_distance = st.slider('Average Transport Distance (km)', min_value=10, max_value=1000, value=250, step=50)
            transport_method = st.selectbox(
                'Primary Transport Method',
                ['Truck', 'Rail', 'Air', 'Ship'],
                index=0
            )
        
        with col2:
            renewable_energy = st.slider('Renewable Energy Usage (%)', min_value=0, max_value=100, value=30, step=10)
            packaging_type = st.selectbox(
                'Primary Packaging Type',
                ['Plastic', 'Paper', 'Biodegradable', 'Reusable'],
                index=0
            )
        
        # Run climate impact analysis button
        if st.button('Analyze Climate Impact'):
            try:
                with st.spinner('Analyzing climate impact...'):
                    # Run analysis
                    climate_impact = climate_analyzer.analyze_impact(
                        forecast_df,
                        optimization_results,
                        transport_distance=transport_distance,
                        transport_method=transport_method,
                        renewable_energy=renewable_energy/100,
                        packaging_type=packaging_type
                    )
                    
                    # Store results
                    st.session_state.climate_impact = climate_impact
                    
                    st.success('Climate impact analysis completed!')
                
                # Display climate impact results
                st.subheader('Carbon Footprint')
                
                # Overall carbon footprint
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total CO2 Emissions", f"{climate_impact['total_emissions']:.2f} tonnes")
                with col2:
                    st.metric("Emissions per Unit", f"{climate_impact['emissions_per_unit']:.2f} kg")
                with col3:
                    st.metric("Carbon Intensity", f"{climate_impact['carbon_intensity']:.2f} kg/$ revenue")
                
                # Emissions breakdown
                st.subheader('Emissions Breakdown')
                emissions_df = pd.DataFrame({
                    'Source': list(climate_impact['emissions_breakdown'].keys()),
                    'CO2 Emissions (tonnes)': list(climate_impact['emissions_breakdown'].values())
                })
                
                fig = px.pie(
                    emissions_df,
                    values='CO2 Emissions (tonnes)',
                    names='Source',
                    title='Carbon Emissions by Source'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Environmental impact timeline
                st.subheader('Environmental Impact Timeline')
                fig = px.line(
                    climate_impact['impact_timeline'],
                    title='Projected Environmental Impact Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Sustainability score
                st.subheader('Sustainability Score')
                
                # Create a gauge chart for sustainability score
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = climate_impact['sustainability_score'],
                    title = {'text': "Sustainability Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Improvement recommendations
                st.subheader('Sustainability Improvement Recommendations')
                for recommendation in climate_impact['recommendations']:
                    st.success(recommendation)
                
                # Impact comparison
                st.subheader('Impact Comparison to Industry Average')
                comparison_df = pd.DataFrame({
                    'Metric': list(climate_impact['industry_comparison'].keys()),
                    'Performance vs Industry Average (%)': list(climate_impact['industry_comparison'].values())
                })
                
                fig = px.bar(
                    comparison_df,
                    x='Metric',
                    y='Performance vs Industry Average (%)',
                    title='Sustainability Performance vs Industry Average',
                    color='Performance vs Industry Average (%)',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error analyzing climate impact: {str(e)}")

# About page
elif page == 'About':
    st.header('About the Food Demand Forecasting System')
    
    st.subheader('System Overview')
    st.write("""
    This application provides end-to-end food demand forecasting and supply chain optimization capabilities. 
    It combines advanced machine learning techniques with supply chain management principles to help food 
    service operations optimize their inventory, reduce waste, and minimize environmental impact.
    """)
    
    st.subheader('Features')
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Processing & Analysis**
        - Historical data processing and cleaning
        - Feature engineering for meal demand prediction
        - Interactive data exploration and visualization
        - Time series analysis and pattern detection
        """)
        
        st.markdown("""
        **Machine Learning Model**
        - Hybrid LSTM + XGBoost architecture
        - Captures both temporal patterns and feature importance
        - Handles multiple centers and meal types
        - Accounts for promotions, pricing, and seasonality
        """)
    
    with col2:
        st.markdown("""
        **Supply Chain Optimization**
        - Inventory requirements calculation with safety buffer
        - Cost-efficient supply chain configuration
        - Waste reduction strategies
        - Lead time and transport efficiency consideration
        """)
        
        st.markdown("""
        **Environmental Impact**
        - Carbon footprint analysis
        - Sustainability scoring
        - Improvement recommendations
        - Comparison to industry standards
        """)
    
    st.subheader('Technologies Used')
    st.markdown("""
    - **Frontend:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Machine Learning:** TensorFlow/Keras, XGBoost, Scikit-learn
    - **Visualization:** Plotly, Matplotlib
    - **Serialization:** Joblib
    """)
    
    st.subheader('How to Use')
    st.markdown("""
    1. **Data Loading & Exploration:** Upload historical data and explore patterns
    2. **Model Training & Forecasting:** Configure and train the hybrid model, then generate forecasts
    3. **Supply Chain Optimization:** Set supply chain parameters and run optimization
    4. **Climate Impact Analysis:** Analyze environmental impact and get sustainability recommendations
    """)
    
    st.info("""
    For best results, provide historical data with at least 50 weeks of demand information per center and 
    meal combination. The system performs better with more historical data and consistent time intervals.
    """)
