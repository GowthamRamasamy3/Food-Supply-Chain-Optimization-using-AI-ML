"""
Utility Functions for Food Supply Optimization
---------------------------------------------
Helper functions for file handling, directory creation, and logging.
"""
import os
import logging
from datetime import datetime

def create_directories():
    """Create necessary directories for data, models, and results."""
    directories = [
        'data/processed',
        'models/full',
        'results/full'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def setup_logging():
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("food_supply_optimization.log"),
            logging.StreamHandler()
        ]
    )
