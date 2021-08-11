import logging
import os


# Non-brain logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=os.environ.get("LOGLEVEL", "INFO"),
)
