import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures a basic but professional logging format for stdout.
    
    Args:
        level: The logging severity level to capture (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Reduce verbose logging from TensorFlow
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
