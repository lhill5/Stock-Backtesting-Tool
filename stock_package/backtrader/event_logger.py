import logging
from global_decorators import *


class Logger:
    def __init__(self, filename_path=None):
        if filename_path is None:
            self.filename_path = 'log_program.log'
        else:
            self.filename_path = filename_path

        self.setup_logger()


    def setup_logger(self):
        LOG_FORMAT = '%(asctime)s\n%(name)s - %(levelname)s - function: %(funcName)s - Line: %(lineno)d - %(message)s'

        file_handler = logging.FileHandler(filename=self.filename_path, mode='w')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger = logging.getLogger(__name__)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

        self.logger = logger


    def get_logger(self):
        return self.logger

