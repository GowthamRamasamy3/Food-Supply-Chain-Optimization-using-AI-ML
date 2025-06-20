"""
Test script for notification system
----------------------------------
This script manually tests the notification system by checking thresholds
and sending emails.
"""
import os
import logging
import pandas as pd
import database
from email_service import process_pending_notifications

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a manual test of the notification system"""
    logger.info("Starting notification test")
    
    # Check SMTP configuration
    smtp_configured = 'SMTP_USERNAME' in os.environ and 'SMTP_PASSWORD' in os.environ
    if not smtp_configured:
        logger.error("SMTP not configured. Set SMTP_USERNAME and SMTP_PASSWORD environment variables.")
        return
    
    # Load predictions
    try:
        results_dir = os.path.join(os.getcwd(), 'results', 'full')
        predictions_file = os.path.join(results_dir, 'food_supply_optimization_predictions_2023.csv')
        
        if not os.path.exists(predictions_file):
            logger.error(f"Predictions file not found: {predictions_file}")
            return
        
        predictions = pd.read_csv(predictions_file)
        logger.info(f"Loaded predictions: {len(predictions)} rows")
        
        # Add a test user if none exists
        test_user = database.get_user_by_email(os.environ.get('SMTP_USERNAME', ''))
        
        if not test_user:
            test_user = database.create_user(
                "Test User",
                os.environ.get('SMTP_USERNAME', '')
            )
            logger.info(f"Created test user: {test_user}")
        else:
            logger.info(f"Using existing user: {test_user}")
        
        if not test_user:
            logger.error("Failed to create or get test user")
            return
        
        # Add test notification preference
        # Get sample center_id and meal_id from predictions
        if len(predictions) > 0:
            sample_row = predictions.iloc[0]
            center_id = sample_row['center_id']
            meal_id = sample_row['meal_id']
            
            # Find a predicted value we can use for testing
            if 'predicted_demand' in predictions.columns:
                pred_col = 'predicted_demand'
            elif 'predicted_orders' in predictions.columns:
                pred_col = 'predicted_orders'
            elif 'prediction' in predictions.columns:
                pred_col = 'prediction'
            else:
                pred_col = predictions.columns[-1]
            
            # Set threshold to be less than the predicted value to trigger the alert
            pred_value = float(sample_row[pred_col])
            threshold_value = pred_value * 0.9  # 90% of the predicted value
            
            # Add the notification preference
            pref = database.add_notification_preference(
                test_user['id'],
                center_id,
                meal_id,
                'above',  # Alert when demand is above the threshold
                threshold_value
            )
            
            if pref:
                logger.info(f"Added test notification preference: {pref}")
            else:
                logger.error("Failed to add notification preference")
                return
            
            # Check thresholds against predictions
            notifications = database.check_thresholds(predictions)
            
            if notifications:
                logger.info(f"Found {len(notifications)} notifications")
                for notif in notifications:
                    logger.info(f"  - {notif['message']}")
                
                # Send notification emails
                sent_count = process_pending_notifications(notifications)
                logger.info(f"Sent {sent_count} notification emails")
            else:
                logger.warning("No notifications triggered")
        else:
            logger.error("Predictions file is empty")
    except Exception as e:
        logger.error(f"Error in notification test: {str(e)}")

if __name__ == "__main__":
    main()