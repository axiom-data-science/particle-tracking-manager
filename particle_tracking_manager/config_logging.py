
import datetime
import logging
from typing import Optional

# def generate_default_output_file():
#     return f"output-results_{datetime.datetime.now():%Y-%m-%dT%H%M%SZ}"


class LoggerMethods:
    """Methods for loggers."""
    
    # def __init__(self):#, log_level: str):
    #     pass

    # def assign_output_file_if_needed(self, value: Optional[str]) -> str: 
    #     if value is None:
    #         value = generate_default_output_file()
    #     return value

    # def clean_output_file(self, value: str) -> str:
    #     value = value.replace(".nc", "").replace(".parquet", "").replace(".parq", "")
    #     return value    

    def close_loggers(self, logger):
        """Close and remove all handlers from the logger."""
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def setup_logger(self, output_file: Optional[str], log_level: str) -> (logging.Logger, str):
        """Setup logger."""

        # output_file = self.assign_output_file_if_needed(output_file)
        # output_file = self.clean_output_file(output_file)
        # # self.output_file = output_file

        logger = logging.getLogger(__package__)
        if logger.handlers:
            self.close_loggers(logger)
            
        logger.setLevel(getattr(logging, log_level))

        # Add handlers from the main logger to the OpenDrift logger if not already added
        
        # Create file handler to save log to file
        logfile_name = pathlib.Path(output_file).stem + ".log"
        file_handler = logging.FileHandler(logfile_name)
        fmt = "%(asctime)s %(levelname)-7s %(name)s.%(module)s.%(funcName)s:%(lineno)d: %(message)s"
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        logger.info("Particle tracking manager simulation.")
        logger.info(f"Output filename: {output_file}")
        logger.info(f"Log filename: {logfile_name}")
        return logger
        # return logger, output_file

    def merge_with_opendrift_log(self, logger: logging.Logger) -> None:
        """Merge the OpenDrift logger with the main logger."""

        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("opendrift"):
                od_logger = logging.getLogger(logger_name)
                if od_logger.handlers:
                    self.close_loggers(od_logger)

                # Add handlers from the main logger to the OpenDrift logger
                for handler in logger.handlers:
                    od_logger.addHandler(handler)
                od_logger.setLevel(logger.level)
                od_logger.propagate = False
