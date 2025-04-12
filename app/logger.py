import logging
from datetime import datetime

logging.basicConfig(
    filename="/var/log/image_processor.log",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)

def log_event(level, message):
    """Log events for Kibana monitoring"""
    timestamp = datetime.now(datetime.UTC).isoformat()
    log_message = f"{timestamp} {level} {message}"
    logging.log(getattr(logging, level.upper()), log_message)
