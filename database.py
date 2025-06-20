"""
Database helper functions for the Food Supply Optimization Dashboard
-------------------------------------------------------------------
Functions to manage database connections, user registration and notification preferences.
"""
import os
import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        # Try to use DATABASE_URL if available
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            conn = psycopg2.connect(
                database_url,
                cursor_factory=RealDictCursor
            )
        else:
            # Otherwise use individual parameters
            conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                database=os.environ.get('DB_NAME', 'food_supply_optimization'),
                user=os.environ.get('DB_USER', 'foodsupply_user'),
                password=os.environ.get('DB_PASSWORD', ''),
                port=os.environ.get('DB_PORT', '5432'),
                cursor_factory=RealDictCursor
            )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return None
def create_user(name, email):
    """
    Create a new user in the database
    
    Args:
        name (str): User's name
        email (str): User's email address
        
    Returns:
        dict: User information if successful, None if failed
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            # Check if user already exists
            cur.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            existing_user = cur.fetchone()
            
            if existing_user:
                return dict(existing_user)
            
            # Insert new user
            cur.execute(
                "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING *",
                (name, email)
            )
            new_user = cur.fetchone()
            conn.commit()
            return dict(new_user)
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_user_by_email(email):
    """
    Get user information by email
    
    Args:
        email (str): User's email address
        
    Returns:
        dict: User information if found, None if not found
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM users WHERE email = %s",
                (email,)
            )
            user = cur.fetchone()
            return dict(user) if user else None
    except Exception as e:
        logger.error(f"Error getting user: {str(e)}")
        return None
    finally:
        conn.close()

def add_notification_preference(user_id, center_id, meal_id, threshold_type, threshold_value):
    """
    Add a new notification preference for a user
    
    Args:
        user_id (int): User ID
        center_id (int): Center ID to monitor
        meal_id (int): Meal ID to monitor
        threshold_type (str): Type of threshold ('above', 'below', 'change_rate')
        threshold_value (float): Value for the threshold
        
    Returns:
        dict: Preference information if successful, None if failed
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Convert numpy types to Python native types
        try:
            user_id = int(user_id)
            center_id = int(center_id)
            meal_id = int(meal_id)
            threshold_value = float(threshold_value)
        except (TypeError, ValueError) as e:
            logger.error(f"Type conversion error: {str(e)}")
            return None
            
        with conn.cursor() as cur:
            # Check if preference already exists
            cur.execute(
                """
                SELECT * FROM notification_preferences 
                WHERE user_id = %s AND center_id = %s AND meal_id = %s AND threshold_type = %s
                """,
                (user_id, center_id, meal_id, threshold_type)
            )
            existing_pref = cur.fetchone()
            
            if existing_pref:
                # Update existing preference
                cur.execute(
                    """
                    UPDATE notification_preferences 
                    SET threshold_value = %s, is_active = TRUE
                    WHERE id = %s RETURNING *
                    """,
                    (threshold_value, existing_pref['id'])
                )
            else:
                # Create new preference
                cur.execute(
                    """
                    INSERT INTO notification_preferences 
                    (user_id, center_id, meal_id, threshold_type, threshold_value)
                    VALUES (%s, %s, %s, %s, %s) RETURNING *
                    """,
                    (user_id, center_id, meal_id, threshold_type, threshold_value)
                )
                
            preference = cur.fetchone()
            conn.commit()
            return dict(preference) if preference else None
    except Exception as e:
        logger.error(f"Error adding notification preference: {str(e)}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_user_preferences(user_id):
    """
    Get all notification preferences for a user
    
    Args:
        user_id (int): User ID
        
    Returns:
        list: List of preference dictionaries
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM notification_preferences WHERE user_id = %s AND is_active = TRUE",
                (user_id,)
            )
            preferences = cur.fetchall()
            return [dict(pref) for pref in preferences]
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        return []
    finally:
        conn.close()

def delete_preference(preference_id, user_id):
    """
    Delete (deactivate) a notification preference
    
    Args:
        preference_id (int): Preference ID to delete
        user_id (int): User ID for verification
        
    Returns:
        bool: True if successful, False if failed
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE notification_preferences 
                SET is_active = FALSE
                WHERE id = %s AND user_id = %s
                """,
                (preference_id, user_id)
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Error deleting preference: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def add_notification(user_id, preference_id, message):
    """
    Add a notification to the queue
    
    Args:
        user_id (int): User ID
        preference_id (int): Preference ID that triggered the notification
        message (str): Notification message
        
    Returns:
        dict: Notification information if successful, None if failed
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO notifications (user_id, preference_id, message) VALUES (%s, %s, %s) RETURNING *",
                (user_id, preference_id, message)
            )
            notification = cur.fetchone()
            conn.commit()
            return dict(notification)
    except Exception as e:
        logger.error(f"Error adding notification: {str(e)}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_pending_notifications():
    """
    Get all pending notifications
    
    Returns:
        list: List of pending notification dictionaries
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT n.*, u.email, u.name
                FROM notifications n
                JOIN users u ON n.user_id = u.id
                WHERE n.is_sent = FALSE
                ORDER BY n.sent_at
                """
            )
            notifications = cur.fetchall()
            return [dict(notif) for notif in notifications]
    except Exception as e:
        logger.error(f"Error getting pending notifications: {str(e)}")
        return []
    finally:
        conn.close()

def mark_notification_sent(notification_id):
    """
    Mark a notification as sent
    
    Args:
        notification_id (int): Notification ID
        
    Returns:
        bool: True if successful, False if failed
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE notifications SET is_sent = TRUE WHERE id = %s",
                (notification_id,)
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Error marking notification sent: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()

def check_thresholds(predictions):
    """
    Check all active notification preferences against predictions
    
    Args:
        predictions (DataFrame): Predictions dataframe
        
    Returns:
        list: List of triggered notifications
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    triggered_notifications = []
    
    try:
        with conn.cursor() as cur:
            # Get all active preferences
            cur.execute(
                """
                SELECT np.*, u.email, u.name
                FROM notification_preferences np
                JOIN users u ON np.user_id = u.id
                WHERE np.is_active = TRUE
                """
            )
            preferences = cur.fetchall()
            
            # Get prediction column name
            pred_col = 'predicted_orders'
            if 'predicted_orders' not in predictions.columns:
                if 'prediction' in predictions.columns:
                    pred_col = 'prediction'
                elif 'predicted_demand' in predictions.columns:
                    pred_col = 'predicted_demand'
                
            # Check each preference against predictions
            for pref in preferences:
                pref_dict = dict(pref)
                filtered_preds = predictions[
                    (predictions['center_id'] == pref_dict['center_id']) &
                    (predictions['meal_id'] == pref_dict['meal_id'])
                ]
                
                if filtered_preds.empty:
                    continue
                
                # Get the latest prediction for this center/meal
                latest_pred = filtered_preds.iloc[-1][pred_col]
                
                # Check threshold
                threshold_triggered = False
                message = ""
                
                if pref_dict['threshold_type'] == 'above' and latest_pred > pref_dict['threshold_value']:
                    threshold_triggered = True
                    message = f"Alert: Demand for meal {pref_dict['meal_id']} at center {pref_dict['center_id']} is above threshold ({latest_pred:.2f} > {pref_dict['threshold_value']:.2f})"
                    
                elif pref_dict['threshold_type'] == 'below' and latest_pred < pref_dict['threshold_value']:
                    threshold_triggered = True
                    message = f"Alert: Demand for meal {pref_dict['meal_id']} at center {pref_dict['center_id']} is below threshold ({latest_pred:.2f} < {pref_dict['threshold_value']:.2f})"
                
                elif pref_dict['threshold_type'] == 'change_rate' and len(filtered_preds) > 1:
                    # Calculate rate of change
                    prev_pred = filtered_preds.iloc[-2][pred_col]
                    change_rate = abs((latest_pred - prev_pred) / prev_pred * 100) if prev_pred != 0 else 0
                    
                    if change_rate > pref_dict['threshold_value']:
                        threshold_triggered = True
                        direction = "increased" if latest_pred > prev_pred else "decreased"
                        message = f"Alert: Demand for meal {pref_dict['meal_id']} at center {pref_dict['center_id']} has {direction} by {change_rate:.2f}% (threshold: {pref_dict['threshold_value']:.2f}%)"
                
                # Create notification if threshold triggered
                if threshold_triggered:
                    notification = add_notification(pref_dict['user_id'], pref_dict['id'], message)
                    if notification:
                        notification.update({
                            'email': pref_dict['email'],
                            'name': pref_dict['name']
                        })
                        triggered_notifications.append(notification)
                        
            return triggered_notifications
    except Exception as e:
        logger.error(f"Error checking thresholds: {str(e)}")
        return []
    finally:
        conn.close()