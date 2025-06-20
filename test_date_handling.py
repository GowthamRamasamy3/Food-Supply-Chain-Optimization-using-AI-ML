"""
Test the date handling in report_generator.py
"""
import os
from datetime import datetime, date
from report_generator import generate_report

def test_date_handling():
    """Test specific date handling to verify fixes"""
    print("\nTesting date handling with explicit dates...")
    
    # Test with a specific date to ensure it doesn't change
    test_date = date(2025, 4, 27)  # April 27, 2025 - should remain April 27 (not change to 28th)
    print(f"Test date object: {test_date} ({type(test_date)})")
    
    # Create date specific data with our test date
    date_specific_data = {
        'forecast_date': test_date,
        'center_id': 20,
        'meal_id': 1993,
        'baseline_demand': 32.72,
        'adjustment_factors': {
            'holiday_factor': 1.0,
            'weather_factor': 1.0,
            'weekday_factor': 1.2,
            'combined_factor': 1.2
        },
        'adjusted_demand': 39.27
    }
    
    # Generate the report with our test date
    output_path = f"./date_test_report_{test_date.strftime('%Y%m%d')}.pdf"
    result = generate_report(output_path, None, None, date_specific_data)
    
    print(f"Generated report: {result}")
    print("Check the PDF to ensure the date is displayed as 'Sunday, April 27, 2025'")
    
    # Test with string date format
    test_date_str = "2025-04-27"  # Should also display as April 27th
    print(f"\nTest date string: {test_date_str}")
    
    date_specific_data['forecast_date'] = test_date_str
    output_path = f"./date_test_report_string.pdf"
    result = generate_report(output_path, None, None, date_specific_data)
    
    print(f"Generated report with string date: {result}")
    print("Check the PDF to ensure the date is displayed as 'Sunday, April 27, 2025'")

if __name__ == "__main__":
    test_date_handling()