import logging
from contextlib import contextmanager


class PocketTTSFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("pocket_tts")


@contextmanager
def enable_logging(library_name, level):
    # Get the specific logger and its parent
    logger = logging.getLogger(library_name)
    parent_logger = logging.getLogger("pocket_tts")

    # Store original configuration
    old_level = logger.level
    old_parent_level = parent_logger.level
    old_handlers = parent_logger.handlers.copy()

    # Configure logging format for pocket_tts logger
    parent_logger.setLevel(level)

    # Clear existing handlers and add our custom formatter with filter
    parent_logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(PocketTTSFilter())
    parent_logger.addHandler(handler)
    parent_logger.propagate = False

    try:
        yield logger
    finally:
        # Restore original configuration
        logger.setLevel(old_level)
        parent_logger.setLevel(old_parent_level)
        parent_logger.handlers.clear()
        for h in old_handlers:
            parent_logger.addHandler(h)
