"""
Email Service for Food Supply Optimization Dashboard
--------------------------------------------------
Functions to send email notifications to users.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_email(recipient_email, recipient_name, subject, message_html):
    """
    Send an email using SMTP
    
    Args:
        recipient_email (str): Recipient's email address
        recipient_name (str): Recipient's name
        subject (str): Email subject
        message_html (str): Email message in HTML format
        
    Returns:
        bool: True if successful, False if failed
    """
    # Get email configuration from environment variables
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_username = os.environ.get('SMTP_USERNAME', 'gowtham3ramasamy@gmail.com')
    smtp_password = os.environ.get('SMTP_PASSWORD', 'Gowtham@k2030')
    sender_email = os.environ.get('SENDER_EMAIL', smtp_username)
    
    # Check if email credentials are configured
    if not smtp_username or not smtp_password:
        logger.error("SMTP credentials not configured")
        return False
    
    try:
        # Create message
        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = f"Food Supply Optimizer <{sender_email}>"
        message['To'] = f"{recipient_name} <{recipient_email}>"
        
        # Create both plain text and HTML versions
        text_content = message_html.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n\n')
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(message_html, 'html')
        
        # Attach parts
        message.attach(text_part)
        message.attach(html_part)
        
        # Connect to server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            
        logger.info(f"Email sent to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

def format_notification_email(name, message):
    """
    Format a notification email
    
    Args:
        name (str): Recipient's name
        message (str): Notification message
        
    Returns:
        str: Formatted HTML email content
    """
    return f"""
    <html>
    <body>
        <h2>Food Supply Optimization Alert</h2>
        <p>Hello {name},</p>
        <p>{message}</p>
        <p>You are receiving this email because you subscribed to alerts for this meal and center.</p>
        <p>To manage your alert preferences, please visit the Food Supply Optimization Dashboard.</p>
        <br>
        <p>Best regards,<br>Food Supply Optimization Team</p>
    </body>
    </html>
    """

def send_notification_email(notification):
    """
    Send a notification email
    
    Args:
        notification (dict): Notification information
        
    Returns:
        bool: True if successful, False if failed
    """
    email = notification.get('email', '')
    name = notification.get('name', '')
    message = notification.get('message', '')
    
    if not email or not message:
        logger.error("Missing email or message for notification")
        return False
    
    subject = "Food Supply Optimization Alert"
    html_content = format_notification_email(name, message)
    
    return send_email(email, name, subject, html_content)

def process_pending_notifications(notifications):
    """
    Process pending notifications
    
    Args:
        notifications (list): List of notification dictionaries
        
    Returns:
        int: Number of successfully sent notifications
    """
    from database import mark_notification_sent
    
    success_count = 0
    
    for notification in notifications:
        if send_notification_email(notification):
            if mark_notification_sent(notification['id']):
                success_count += 1
    
    return success_count