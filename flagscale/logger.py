import logging
import sys


class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s] %(message)s"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def debug(self, message):
        self.logger.debug(message)


# Global registry for multiple logger instances
_GLOBAL_LOGGERS = {}


def get_logger(name="FlagScale"):
    """Get a singleton Logger instance by name."""
    if name not in _GLOBAL_LOGGERS:
        _GLOBAL_LOGGERS[name] = Logger(name)
    return _GLOBAL_LOGGERS[name]


# Default main logger
logger = get_logger("FlagScale")

# Dedicated loggers for submodules
train_logger = get_logger("FlagScale.train")
inference_logger = get_logger("FlagScale.inference")
compress_logger = get_logger("FlagScale.compress")
serve_logger = get_logger("FlagScale.serve")
rl_logger = get_logger("FlagScale.rl")
