"""
Food Supply Optimization Model
------------------------------
Main script to run the food supply optimization model which predicts demand
for meals across multiple centers based on various factors.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import flask
from flask import Flask, render_template, jsonify, request, Response
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


from data_processing import (
    load_or_generate_data,
    preprocess_data,
    get_train_test_data,
    predict_for_specific_date
)
from model import (
    load_or_train_model,
    evaluate_model,
    predict_future_demand
)
from utils import create_directories, setup_logging

# Setup logging
setup_logging()

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main dashboard page"""
    # Get today's date as default
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Return rendered template with tomorrow's date
    return render_template('index.html', tomorrow=tomorrow)

@app.route('/run-model')
def run_model():
    """Run the optimization model and return the results"""
    try:
        output = []
        output.append("# Optimizing Food Supply Model #")
        
        # Get parameters from request
        year = request.args.get('year', default='2023', type=str)
        force_regenerate = request.args.get('force_regenerate', default='false', type=str)
        
        # Convert parameters
        force_regenerate = True if force_regenerate.lower() == 'true' else False
        
        try:
            year_int = int(year)
            if year_int < 2020 or year_int > 2050:
                raise ValueError("Year must be between 2020 and 2050")
        except ValueError:
            year_int = 2023
        
        # Load or generate data
        train_data, test_data = load_or_generate_data(base_year=year_int, force_regenerate=force_regenerate)
        
        # Process data
        X_train, y_train, X_val, y_val, X_test = get_train_test_data(train_data, test_data)
        
        # Load or train models
        lstm_model, xgb_model = load_or_train_model(X_train, y_train)
        
        # Evaluate model
        val_preds = evaluate_model(lstm_model, xgb_model, X_val, y_val)
        
        # Make future predictions
        predictions = predict_future_demand(lstm_model, xgb_model, X_test, test_data)
        
        # Log success
        output.append("Food supply optimization model training completed successfully!")
        
        return jsonify({
            'success': True,
            'output': '\n'.join(output)
        })
    except Exception as e:
        logging.error(f"Error in run_model: {str(e)}")
        return jsonify({
            'success': False,
            'output': f"Error: {str(e)}"
        })

@app.route('/predict-specific')
def predict_specific():
    """Generate predictions for a specific date and city"""
    try:
        # Get parameters from request
        date_str = request.args.get('date', default='', type=str)
        city = request.args.get('city', default='', type=str)
        model_year = request.args.get('model_year', default='2025', type=str)
        
        # Validate parameters
        if not date_str:
            return jsonify({'success': False, 'error': 'Date is required'})
        
        if not city:
            return jsonify({'success': False, 'error': 'City is required'})
        
        # Convert model year
        try:
            model_year_int = int(model_year)
            if model_year_int < 2020 or model_year_int > 2050:
                return jsonify({'success': False, 'error': 'Model year must be between 2020 and 2050'})
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid model year'})
        
        # Generate predictions
        predictions, context = predict_for_specific_date(date_str, city, model_year=model_year_int)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'context': context
        })
    except Exception as e:
        logging.error(f"Error in predict_specific: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/visualize/climate-impact')
def visualize_climate_impact():
    """Generate and return climate impact visualization"""
    try:
        # Get selected year
        year = request.args.get('year', default='2025', type=str)
        
        try:
            year_int = int(year)
            if year_int < 2020 or year_int > 2050:
                raise ValueError("Year must be between 2020 and 2050")
        except ValueError:
            year_int = 2025
        
        # Generate data for different climate scenarios
        base_temp_increase = (year_int - 2020) * 0.03  # approx 0.03C per year increase
        
        # Temperature ranges from -0.5C to +2.0C of the projected increase
        temp_scenarios = np.linspace(base_temp_increase - 0.5, base_temp_increase + 2.0, 6)
        
        # Demand factors - made up for demonstration
        # In reality, this would come from the model
        demand_impact = np.array([0.95, 0.98, 1.0, 1.05, 1.12, 1.18])
        
        # Create the figure
        plt.figure(figsize=(10, 5))
        plt.plot(temp_scenarios, demand_impact, 'o-', linewidth=2)
        plt.xlabel(f'Temperature Increase from 2020 (C) - Year {year_int}')
        plt.ylabel('Demand Impact Factor')
        plt.title(f'Climate Impact on Food Demand - {year_int}')
        plt.grid(True, alpha=0.3)
        
        # Add current year marker
        plt.axvline(x=base_temp_increase, color='r', linestyle='--', alpha=0.7)
        plt.annotate(f'Current ({year_int})', 
                     xy=(base_temp_increase, 1.0),
                     xytext=(base_temp_increase + 0.2, 1.02),
                     arrowprops=dict(arrowstyle='->'))
        
        # Add a colored band for critical regions
        plt.axhspan(0.95, 1.0, alpha=0.2, color='blue', label='Decreased Demand')
        plt.axhspan(1.0, 1.2, alpha=0.2, color='red', label='Increased Demand')
        
        plt.legend()
        
        # Save figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Return the image as a response
        return Response(buf.getvalue(), mimetype='image/png')
    except Exception as e:
        logging.error(f"Error in visualize_climate_impact: {str(e)}")
        # Return an error image
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return Response(buf.getvalue(), mimetype='image/png')

@app.route('/visualize/demand-forecast')
def visualize_demand_forecast():
    """Generate and return demand forecast by country visualization"""
    try:
        # Get selected year
        year = request.args.get('year', default='2025', type=str)
        
        try:
            year_int = int(year)
            if year_int < 2020 or year_int > 2050:
                raise ValueError("Year must be between 2020 and 2050")
        except ValueError:
            year_int = 2025
        
        # Sample data - in reality would come from predictions
        countries = ['India', 'USA', 'UK']
        base_demand = np.array([1200, 900, 700])
        climate_factor = 1.0 + (year_int - 2020) * 0.02
        
        # Apply different growth rates for different countries
        demand_forecast = base_demand * climate_factor * np.array([1.03, 1.02, 1.01]) ** (year_int - 2020)
        
        # Create figure
        plt.figure(figsize=(10, 5))
        bars = plt.bar(countries, demand_forecast, color=['#FF9999', '#66B2FF', '#99FF99'])
        
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{int(height)}',
                    ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Country')
        plt.ylabel('Predicted Daily Demand (meals)')
        plt.title(f'Forecasted Daily Meal Demand by Country - {year_int}')
        plt.ylim(0, max(demand_forecast) * 1.2)  # Add some space for the text
        
        # Save figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Return the image as a response
        return Response(buf.getvalue(), mimetype='image/png')
    except Exception as e:
        logging.error(f"Error in visualize_demand_forecast: {str(e)}")
        # Return an error image
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return Response(buf.getvalue(), mimetype='image/png')

@app.route('/visualize/meal-popularity')
def visualize_meal_popularity():
    """Generate and return meal popularity visualization"""
    try:
        # Get selected year
        year = request.args.get('year', default='2025', type=str)
        
        try:
            year_int = int(year)
            if year_int < 2020 or year_int > 2050:
                raise ValueError("Year must be between 2020 and 2050")
        except ValueError:
            year_int = 2025
            
        # Sample data for meal popularity
        meal_names = ['Butter Chicken', 'Pasta Carbonara', 'Vegetable Curry', 
                      'Fish & Chips', 'Tacos', 'Sushi Platter']
        
        # Generate semi-realistic demand numbers with random variations
        base_values = np.array([320, 280, 230, 190, 310, 270])
        seasonal_factor = 1.0 + 0.15 * np.sin(np.pi * (year_int - 2020) / 15)
        demand_values = base_values * seasonal_factor
        
        # Add some random variation to make it look more natural
        np.random.seed(year_int)  # Seed for reproducibility
        demand_values = demand_values * (1 + np.random.normal(0, 0.05, size=len(base_values)))
        
        # Sort by popularity
        sorted_indices = np.argsort(demand_values)[::-1]  # Descending order
        meals_sorted = [meal_names[i] for i in sorted_indices]
        demand_sorted = demand_values[sorted_indices]
        
        # Create horizontal bar chart for meal popularity
        plt.figure(figsize=(10, 6))
        bars = plt.barh(meals_sorted, demand_sorted, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(meals_sorted))))
        
        # Add values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}',
                    ha='left', va='center')
        
        plt.xlabel('Average Daily Orders')
        plt.title(f'Meal Popularity Ranking - {year_int}')
        plt.tight_layout()
        
        # Save figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Return the image as a response
        return Response(buf.getvalue(), mimetype='image/png')
    except Exception as e:
        logging.error(f"Error in visualize_meal_popularity: {str(e)}")
        # Return an error image
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return Response(buf.getvalue(), mimetype='image/png')

def main():
    """Main function to run the food supply optimization model."""
    # Create necessary directories
    create_directories()
    
    # Load or generate data with base_year=2025 (future climate projection)
    train_data, test_data = load_or_generate_data(base_year=2025)
    
    # Process data for model training
    X_train, y_train, X_val, y_val, X_test = get_train_test_data(train_data, test_data)
    
    # Load or train models
    lstm_model, xgb_model = load_or_train_model(X_train, y_train)
    
    # Evaluate model on validation data
    val_preds = evaluate_model(lstm_model, xgb_model, X_val, y_val)
    
    # Make future predictions
    predictions = predict_future_demand(lstm_model, xgb_model, X_test, test_data)
    
    logging.info("Food supply optimization model training completed successfully!")
    
if __name__ == "__main__":
    main()