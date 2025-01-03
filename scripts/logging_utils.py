import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs', log_file=None, log_level=logging.INFO):
    """
    Set up the logger for the project.
    
    Args:
        log_dir (str): Directory where the log files will be saved.
        log_file (str): Name of the log file. Defaults to timestamped file.
        log_level (int): Logging level. Defaults to logging.INFO.
    
    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create log directory if it doesn't exist

    # Default log file name if not provided
    if log_file is None:
        log_file = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    log_path = os.path.join(log_dir, log_file)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),  
            logging.StreamHandler()      
        ]
    )

    logger = logging.getLogger(__name__)
    return logger
