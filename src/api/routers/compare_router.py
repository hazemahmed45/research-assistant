"""
Compare Document API Module

This module defines the API endpoint for compare document.
It provides an endpoint to compare documents in the request body with each others

"""

from __future__ import annotations
import sys
from fastapi import APIRouter, BackgroundTasks
import loguru
from loguru import logger

from src.api.config import ApiSettings
from src.api.schema import (
    DocumentsComparisonOutputSchema,
    DocumentsComparisonInputSchema,
)
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id
from src.enums import Constants

# REQUEST_TYPE = "comp-doc"

settings = ApiSettings()
comp_doc_router = APIRouter()


@comp_doc_router.post("/comp_doc", tags=["Compare Document"])
async def compare_doc(
    schema: DocumentsComparisonInputSchema,
    background_tasks: BackgroundTasks,
) -> DocumentsComparisonOutputSchema:
    """
    TODO docstring
    """
    request_unique_id: str = create_unique_user_id()
    session_logger: loguru.Logger = logger.bind(
        user_unique_id=request_unique_id,
        request_type=Constants.COMPARE_DOCUMENTS_REQUEST_TYPE.value,
    )
    session_logger.remove()
    session_logger.add(
        sys.stderr,
        format=Constants.LOGGER_REQUEST_FORMAT.value,
    )

    session_logger.add(
        sink=FileHandler(user_unique_id=request_unique_id),
        format=Constants.LOGGER_REQUEST_FORMAT.value,
    )

    session_logger.info("Request Recieved")

    # TODO IMPLEMENT COMPARE DOCUMENTS ROUTE
    background_tasks.add_task(session_logger.remove)
    return DocumentsComparisonOutputSchema()
