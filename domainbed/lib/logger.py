import logging
import sys

def create_logger(name, level="info"):
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    logger = logging.getLogger(name)
    logger.setLevel(level_map.get(level.lower(), logging.INFO))
    
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(level_map.get(level.lower(), logging.INFO))
    
    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    
    # Add handlers to the logger if they haven't been added yet
    if not logger.handlers:
        logger.addHandler(c_handler)
    
    return logger