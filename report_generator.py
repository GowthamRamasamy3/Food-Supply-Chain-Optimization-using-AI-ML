"""
PDF Report Generator for Food Supply Optimization Dashboard
---------------------------------------------------------
Generates comprehensive PDF reports with forecasts, visualizations, and optimization results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus import PageBreak
from reportlab.lib.units import inch

def create_forecast_plot(predictions, center_id, meal_id, pred_col):
    """
    Create a forecast plot for a specific center and meal.
    
    Args:
        predictions (DataFrame): Forecast predictions
        center_id (int): Center ID
        meal_id (int): Meal ID
        pred_col (str): Column name for predictions
        
    Returns:
        BytesIO: Plot image as bytes
    """
    # Filter data
    center_meal_forecast = predictions[
        (predictions['center_id'] == center_id) &
        (predictions['meal_id'] == meal_id)
    ]
    
    if center_meal_forecast.empty:
        return None
    
    # Create plot
    plt.figure(figsize=(8, 4))
    plt.plot(center_meal_forecast['week'], center_meal_forecast[pred_col], marker='o')
    plt.title(f'Demand Forecast for Center {center_id}, Meal {meal_id}')
    plt.xlabel('Week')
    plt.ylabel('Predicted Orders')
    plt.grid(linestyle='--', alpha=0.7)
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def create_optimization_plot(inventory_df, center_id, meal_id, pred_col):
    """
    Create an optimization plot for a specific center and meal.
    
    Args:
        inventory_df (DataFrame): Inventory optimization results
        center_id (int): Center ID
        meal_id (int): Meal ID
        pred_col (str): Column name for predictions
        
    Returns:
        BytesIO: Plot image as bytes
    """
    # Filter data
    center_meal_inventory = inventory_df[
        (inventory_df['center_id'] == center_id) &
        (inventory_df['meal_id'] == meal_id)
    ]
    
    if center_meal_inventory.empty:
        return None
    
    # Create plot
    plt.figure(figsize=(8, 4))
    plt.plot(center_meal_inventory['week'], center_meal_inventory[pred_col], marker='o', label='Predicted Demand')
    plt.plot(center_meal_inventory['week'], center_meal_inventory['final_inventory'], marker='s', label='Required Inventory')
    plt.title(f'Inventory Plan for Center {center_id}, Meal {meal_id}')
    plt.xlabel('Week')
    plt.ylabel('Units')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def create_pie_chart(inventory_df):
    """
    Create a pie chart for cost distribution.
    
    Args:
        inventory_df (DataFrame): Inventory optimization results
        
    Returns:
        BytesIO: Plot image as bytes
    """
    # Calculate costs
    storage_cost = inventory_df['storage_cost'].sum()
    transport_cost = inventory_df['transport_cost'].sum()
    shortage_cost = inventory_df['shortage_cost'].sum()
    
    # Create plot
    plt.figure(figsize=(6, 6))
    plt.pie(
        [storage_cost, transport_cost, shortage_cost],
        labels=['Storage', 'Transport', 'Shortage'],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=(0.05, 0, 0)
    )
    plt.title('Cost Distribution')
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def create_date_specific_plot(demand_factors):
    """
    Create a bar chart for date-specific adjustment factors.
    
    Args:
        demand_factors (dict): Demand adjustment factors
        
    Returns:
        BytesIO: Plot image as bytes
    """
    # Extract factors
    factors = [
        (demand_factors['holiday_factor'] - 1) * 100,
        (demand_factors['weather_factor'] - 1) * 100,
        (demand_factors['weekday_factor'] - 1) * 100,
        (demand_factors['combined_factor'] - 1) * 100
    ]
    
    # Create plot
    plt.figure(figsize=(8, 4))
    bars = plt.bar(
        ['Holiday', 'Weather', 'Day of Week', 'Combined'],
        factors,
        color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.3,
            f'{height:.1f}%',
            ha='center'
        )
    
    plt.title('Demand Adjustment Factors')
    plt.ylabel('Impact (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def dataframe_to_table(df, max_rows=None):
    """
    Convert a pandas DataFrame to a ReportLab Table.
    
    Args:
        df (DataFrame): DataFrame to convert
        max_rows (int, optional): Maximum number of rows to include
        
    Returns:
        tuple: (Table, TableStyle)
    """
    # Limit rows if needed
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    
    # Convert to list of lists
    data = [df.columns.tolist()]
    data.extend(df.values.tolist())
    
    # Create table style
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ])
    
    # Create table
    table = Table(data)
    table.setStyle(style)
    
    return table

def generate_full_report(
    predictions, 
    inventory_df=None, 
    service_level=None, 
    center_id=None, 
    meal_id=None,
    date_specific=None,
    output_path=None
):
    """
    Generate a comprehensive PDF report with forecasts and optimization results.
    
    Args:
        predictions (DataFrame): Forecast predictions
        inventory_df (DataFrame, optional): Inventory optimization results
        service_level (float, optional): Service level from optimization
        center_id (int, optional): Center ID to focus on
        meal_id (int, optional): Meal ID to focus on
        date_specific (dict, optional): Date-specific forecast data
        output_path (str, optional): Path to save the PDF file
        
    Returns:
        str: Path to the generated PDF file
    """
    # Set default output path if not provided
    if output_path is None:
        output_path = f"food_supply_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Determine the column name for predictions
    if 'predicted_orders' in predictions.columns:
        pred_col = 'predicted_orders'
    elif 'prediction' in predictions.columns:
        pred_col = 'prediction'
    else:
        pred_col = predictions.columns[-1]  # Fallback to last column
    
    # Select center and meal if not provided
    if center_id is None:
        center_id = predictions['center_id'].min()
    if meal_id is None:
        meal_id = predictions['meal_id'].min()
    
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create story
    story = []
    
    # Title
    story.append(Paragraph("Food Supply Optimization Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Forecast section
    story.append(Paragraph("Demand Forecast", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Forecast summary
    summary_text = f"""
    This report provides a comprehensive analysis of demand forecasts and supply chain optimization
    for food distribution centers. The forecast covers {len(predictions['week'].unique())} weeks, 
    {len(predictions['center_id'].unique())} centers, and {len(predictions['meal_id'].unique())} meal types.
    """
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Forecast table
    story.append(Paragraph("Forecast Sample Data", subheading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Create table for forecast data (first 10 rows)
    forecast_sample = predictions.head(10).reset_index(drop=True)
    forecast_table = dataframe_to_table(forecast_sample)
    story.append(forecast_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Forecast visualization
    story.append(Paragraph("Forecast Visualization", subheading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Create forecast plot for the selected center and meal
    plot_bytes = create_forecast_plot(predictions, center_id, meal_id, pred_col)
    if plot_bytes:
        story.append(Image(plot_bytes, width=6.5*inch, height=3.5*inch))
    else:
        story.append(Paragraph("No forecast data available for the selected center and meal combination.", normal_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Add page break after forecast section
    story.append(PageBreak())
    
    # Optimization section
    if inventory_df is not None:
        story.append(Paragraph("Supply Chain Optimization", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Optimization summary
        if service_level is not None:
            total_inventory = inventory_df['final_inventory'].sum()
            total_cost = inventory_df['total_cost'].sum()
            avg_weekly_cost = total_cost / len(inventory_df['week'].unique())
            
            summary_text = f"""
            The supply chain optimization achieves a service level of {service_level:.2%} with
            a total required inventory of {int(total_inventory):,} units. The total cost is 
            ${total_cost:,.2f} with an average weekly cost of ${avg_weekly_cost:,.2f}.
            """
            story.append(Paragraph(summary_text, normal_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Cost breakdown
        story.append(Paragraph("Cost Distribution", subheading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Create pie chart for cost distribution
        pie_bytes = create_pie_chart(inventory_df)
        story.append(Image(pie_bytes, width=4*inch, height=4*inch))
        story.append(Spacer(1, 0.2*inch))
        
        # Inventory planning
        story.append(Paragraph(f"Inventory Plan for Center {center_id}, Meal {meal_id}", subheading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Create optimization plot
        opt_plot_bytes = create_optimization_plot(inventory_df, center_id, meal_id, pred_col)
        if opt_plot_bytes:
            story.append(Image(opt_plot_bytes, width=6.5*inch, height=3.5*inch))
        else:
            story.append(Paragraph("No inventory data available for the selected center and meal combination.", normal_style))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed inventory table
        story.append(Paragraph("Detailed Inventory Plan (Sample)", subheading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Filter data for the selected center and meal
        center_meal_inventory = inventory_df[
            (inventory_df['center_id'] == center_id) &
            (inventory_df['meal_id'] == meal_id)
        ]
        
        if not center_meal_inventory.empty:
            # Select relevant columns
            display_cols = ['week', pred_col, 'base_inventory', 'lead_time_demand', 'final_inventory', 'total_cost']
            # Remove duplicates
            unique_cols = []
            for col in display_cols:
                if col not in unique_cols and col in center_meal_inventory.columns:
                    unique_cols.append(col)
            
            # Create table
            inv_sample = center_meal_inventory[unique_cols].head(10).reset_index(drop=True)
            inv_table = dataframe_to_table(inv_sample)
            story.append(inv_table)
        else:
            story.append(Paragraph("No inventory data available for the selected center and meal combination.", normal_style))
        
        # Add page break after optimization section
        story.append(PageBreak())
    
    # Date-specific forecast section
    if date_specific is not None:
        date_str = date_specific.get('date_str', 'Selected Date')
        baseline_demand = date_specific.get('baseline_demand', 0)
        adjusted_demand = date_specific.get('adjusted_demand', 0)
        combined_factor = date_specific.get('combined_factor', 1)
        demand_factors = date_specific.get('demand_factors', {})
        
        story.append(Paragraph("Date-Specific Forecast", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Summary
        summary_text = f"""
        Forecast for {date_str} (Center {center_id}, Meal {meal_id})
        
        Baseline Demand: {baseline_demand:.2f} units
        Adjustment Factor: {combined_factor:.2f}x
        Adjusted Demand: {adjusted_demand:.2f} units ({(combined_factor-1)*100:.1f}%)
        """
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 0.15*inch))
        
        # Factors visualization
        if demand_factors:
            story.append(Paragraph("Demand Adjustment Factors", subheading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Create factor plot
            factor_plot = create_date_specific_plot(demand_factors)
            story.append(Image(factor_plot, width=6.5*inch, height=3.5*inch))
            
            # Add factors table
            story.append(Spacer(1, 0.15*inch))
            story.append(Paragraph("Adjustment Factors Breakdown", subheading_style))
            story.append(Spacer(1, 0.1*inch))
            
            factor_data = pd.DataFrame({
                'Factor': ['Holiday', 'Weather', 'Day of Week', 'Combined'],
                'Impact (%)': [
                    f"{(demand_factors['holiday_factor']-1)*100:.1f}%", 
                    f"{(demand_factors['weather_factor']-1)*100:.1f}%",
                    f"{(demand_factors['weekday_factor']-1)*100:.1f}%",
                    f"{(demand_factors['combined_factor']-1)*100:.1f}%"
                ],
                'Multiplier': [
                    f"{demand_factors['holiday_factor']:.2f}x",
                    f"{demand_factors['weather_factor']:.2f}x",
                    f"{demand_factors['weekday_factor']:.2f}x",
                    f"{demand_factors['combined_factor']:.2f}x"
                ]
            })
            
            factor_table = dataframe_to_table(factor_data)
            story.append(factor_table)
            
            # Holiday information
            holiday_name = demand_factors.get('holiday', 'No Holiday')
            if holiday_name != 'No Holiday':
                story.append(Spacer(1, 0.15*inch))
                story.append(Paragraph(f"Note: Due to {holiday_name}, special promotional offers may increase demand beyond the forecast.", normal_style))
    
    # Build the PDF
    doc.build(story)
    
    return output_path

def generate_report_from_session(session_state, center_id=None, meal_id=None, output_path=None):
    """
    Generate a comprehensive PDF report from the session state.
    
    Args:
        session_state (dict): Streamlit session state
        center_id (int, optional): Center ID to focus on
        meal_id (int, optional): Meal ID to focus on
        output_path (str, optional): Path to save the PDF file
        
    Returns:
        str: Path to the generated PDF file
    """
    # Check for required data
    if 'predictions' not in session_state or not session_state.predictions_generated:
        raise ValueError("Forecast predictions not available. Please generate forecast first.")
    
    # Prepare date-specific data if available
    date_specific = None
    if hasattr(session_state, 'date_specific_forecast') and session_state.date_specific_forecast:
        date_specific = session_state.date_specific_forecast
    
    # Generate the report
    return generate_full_report(
        predictions=session_state.predictions,
        inventory_df=session_state.inventory_df if hasattr(session_state, 'inventory_df') else None,
        service_level=session_state.service_level if hasattr(session_state, 'service_level') else None,
        center_id=center_id,
        meal_id=meal_id,
        date_specific=date_specific,
        output_path=output_path
    )