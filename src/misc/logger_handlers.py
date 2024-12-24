"""
Custom File Handler for Loguru Logger
=====================================

This module provides a custom file handler for the Loguru logger, allowing for logging to a file system.
"""

from __future__ import annotations
from typing import List
from logging import Handler, LogRecord
import fsspec


class FileHandler(Handler):
    """
    **Loguru File Handler for User-Specific Logging**

    This class extends the Loguru Handler to write logs to a file system, with support for user-specific logging.

    """

    def __init__(self, user_unique_id: str, logs_dir="./logs"):
        """
        **Initialize the File Handler**

        :param user_unique_id: Unique identifier for the user, used for logging separation
        :type user_unique_id: str
        :param logs_dir: Directory for storing log files, defaults to "./logs"
        :type logs_dir: str, optional
        """
        super(FileHandler, self).__init__()
        self.user_unique_id = user_unique_id
        self.log_dir = logs_dir
        self.fs: fsspec.AbstractFileSystem = fsspec.core.url_to_fs(self.log_dir)[0]
        if not self.fs.exists(self.log_dir):
            self.fs.makedirs(self.log_dir)
        self.log_filepath = f"{self.log_dir}{self.fs.sep}{self.user_unique_id}.log"

    def emit(self, record: LogRecord) -> None:
        """
        **Write a Log Record to the File**

        This method writes a log record to the file, ensuring the user unique ID matches the handler's ID.

        :param record: The log record to write
        :type record: LogRecord
        :raises Exception: If the user unique ID is not present in the record or does not match the handler's ID
        """
        if "user_unique_id" not in record.extra.keys():
            raise Exception(
                'did not pass unique user id to handler to log seperate request, pass the "user_unique_id" as an argument in the logger.log or logger.info'
            )
        user_unique_id = record.extra["user_unique_id"]
        if self.user_unique_id == user_unique_id:
            with fsspec.open(self.log_filepath, mode="a") as f:
                f.writelines(record.msg + "\n")
            return
