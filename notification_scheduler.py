"""
Notification Scheduler for Food Supply Optimization Dashboard
------------------------------------------------------------
Scheduled job to check for notifications and send emails.
"""
import os
import time
import logging
import pandas as pd
import threading
from datetime import datetime

import database
from email_service import process_pending_notifications

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_latest_predictions():
    """
    Load the latest predictions
    
    Returns:
        DataFrame: Latest prediction data
    """
    try:
        results_dir = os.path.join(os.getcwd(), 'results', 'full')
        
        # Try 2023 predictions first, then 2025, then the default
        for year in ['2023', '2025', '']:
            file_suffix = f"_{year}" if year else ""
            predictions_file = os.path.join(
                results_dir, 
                f'food_supply_optimization_predictions{file_suffix}.csv'
            )
            
            if os.path.exists(predictions_file):
                logger.info(f"Loading predictions from {predictions_file}")
                return pd.read_csv(predictions_file)
        
        logger.error("No prediction files found")
        return None
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return None

def check_for_notifications():
    """
    Check for notifications and send emails
    
    Returns:
        int: Number of notifications processed
    """
    # Load predictions
    predictions = load_latest_predictions()
    if predictions is None:
        logger.error("Failed to load predictions")
        return 0
    
    # Check thresholds against predictions
    triggered_notifications = database.check_thresholds(predictions)
    if not triggered_notifications:
        logger.info("No notification thresholds were triggered")
        return 0
    
    # Send emails for triggered notifications
    logger.info(f"Found {len(triggered_notifications)} notifications to send")
    sent_count = process_pending_notifications(triggered_notifications)
    
    logger.info(f"Successfully sent {sent_count} notification emails")
    return sent_count

def notification_scheduler(interval_minutes=60):
    """
    Run the notification scheduler in a loop
    
    Args:
        interval_minutes (int): Interval between checks in minutes
    """
    logger.info(f"Starting notification scheduler with {interval_minutes} minute interval")
    
    while True:
        logger.info("Running scheduled notification check...")
        
        try:
            notifications_sent = check_for_notifications()
            logger.info(f"Notification check complete. Sent {notifications_sent} notifications.")
        except Exception as e:
            logger.error(f"Error in notification check: {str(e)}")
        
        # Sleep until next check
        logger.info(f"Next check in {interval_minutes} minutes")
        time.sleep(interval_minutes * 60)

def start_scheduler_thread(interval_minutes=60):
    """
    Start the scheduler in a background thread
    
    Args:
        interval_minutes (int): Interval between checks in minutes
    """
    scheduler_thread = threading.Thread(
        target=notification_scheduler,
        args=(interval_minutes,),
        daemon=True  # Allow the thread to exit when the main process exits
    )
    scheduler_thread.start()
    logger.info(f"Notification scheduler thread started with {interval_minutes} minute interval")
    return scheduler_thread

if __name__ == "__main__":
    # When run directly, check notifications immediately
    notifications_sent = check_for_notifications()
    print(f"Sent {notifications_sent} notifications")