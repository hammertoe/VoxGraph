# -*- coding: utf-8 -*-
import logging
import sys
import threading

def setup_logging():
    """Configures the application's root logger."""
    logger = logging.getLogger("VoxGraph") # Use a specific name for your app's logger
    if not logger.handlers: # Avoid adding handlers multiple times
        logger.setLevel(logging.INFO) # Set desired level
        stream_handler = logging.StreamHandler(sys.stdout) # Use stdout
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s [%(threadName)s] %(levelname)s - %(message)s'
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False # Prevent duplication if root logger also has handlers
        logger.info("--- VoxGraph Logger Initialized ---")
    else:
        logger.info("--- VoxGraph Logger Already Initialized ---")
    return logger

def get_log_prefix(session_id: str | None = None, sid: str | None = None, component: str | None = None) -> str:
    """Creates a standardized log prefix string."""
    parts = []
    if session_id:
        parts.append(f"Session:{session_id[:6]}")
    if sid:
        parts.append(f"SID:{sid[:6]}")
    if component:
        parts.append(component)

    if not parts:
        return "[System]"
    else:
        return f"[{'|'.join(parts)}]"

# Initialize logger when module is loaded
logger = setup_logging()

# Configure NLTK logger to use our handler (optional, can be noisy)
# try:
#     nltk_logger = logging.getLogger('nltk')
#     if not nltk_logger.handlers:
#         nltk_logger.setLevel(logging.INFO) # Or WARNING
#         for handler in logger.handlers:
#             nltk_logger.addHandler(handler)
#     nltk_logger.propagate = False
# except Exception as e:
#     logger.warning(f"Could not configure NLTK logger: {e}")
