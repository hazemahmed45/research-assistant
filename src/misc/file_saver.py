"""
File Saver Classes
==================

This module provides classes for saving files to various destinations, including local file systems.
"""

from __future__ import annotations
import shutil
from typing import Any, Dict
import os
import json
import numpy as np
import pydub
import fsspec
import fsspec.generic
import loguru


class FileSaver:
    """
    **Abstract Base Class for File Savers**

    This class provides a basic interface for file savers, including logging and S3 client setup.
    """

    def __init__(self, logger: loguru.Logger = None) -> None:
        """
        **Initialize the File Saver**

        :param logger: Optional logger instance, defaults to None
        :type logger: loguru.Logger, optional
        """
        self.format = ""
        self.logger = logger

    def write_file(self, dest_filepath: str, *args, **kwargs):
        """
        **Write a File to the Destination**

        This method must be implemented by subclasses.

        :param dest_filepath: Destination file path
        :type dest_filepath: str
        :raises NotImplementedError: Always raised, as this method is abstract
        """
        raise NotImplementedError("generic file saver is not implemented")


class JsonFileSaver(FileSaver):
    """
    **JSON File Saver**

    This class saves JSON data to the destination, with support for local file systems and AWS S3.
    """

    def __init__(self, logger: loguru.Logger = None) -> None:
        """
        **Initialize the JSON File Saver**

        :param logger: Optional logger instance, defaults to None
        :type logger: loguru.Logger, optional
        """
        super().__init__(logger=logger)
        self.format = "json"

    def write_file(
        self,
        dest_filepath: str,
        json_obj: Dict,
        *args,
        **kwargs,
    ):
        """
        **Write JSON Data to the Destination**

        :param dest_filepath: Destination file path
        :type dest_filepath: str
        :param json_obj: JSON data as a dictionary
        :type json_obj: Dict
        """
        if self.logger is not None:
            self.logger.info("Start uploading json file")
        if self.format not in dest_filepath:
            dest_filepath = dest_filepath + f".{self.format}"
        with fsspec.open(dest_filepath, mode="w") as f:
            json.dump(json_obj, fp=f)
        if self.logger is not None:
            self.logger.info("Finish uploading json file")
        return


class PdfFileSaver(FileSaver):
    def __init__(self, logger=None):
        super().__init__(logger)

    def write_file(self, dest_filepath, *args, **kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    pass
