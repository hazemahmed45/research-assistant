"""
Similar Document API Module

This module defines the API endpoint for similar document.
It provides an endpoint to retrieve similar documents to the given document in the request body

"""

from __future__ import annotations
import sys
from fastapi import APIRouter, BackgroundTasks
import loguru
from loguru import logger

from src.api.config import ApiSettings
from src.api.schema import (
    DocumentsSimilarityOutputSchema,
    DocumentsComparisonInputSchema,
)
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id
from src.enums import Constants

# REQUEST_TYPE = "sim-doc"

settings = ApiSettings()
sim_doc_router = APIRouter()


@sim_doc_router.post("/sim_doc", tags=["Similar Document"])
async def similar_doc(
    schema: DocumentsComparisonInputSchema,
    background_tasks: BackgroundTasks,
) -> DocumentsSimilarityOutputSchema:
    """
    TODO docstring
    """
    request_unique_id: str = create_unique_user_id()
    session_logger: loguru.Logger = logger.bind(
        user_unique_id=request_unique_id,
        request_type=Constants.SIMILAR_DOCUMENT_REQUEST_TYPE.value,
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

    # TODO IMPLEMENT SIMILAR DOCUMENTS ROUTE
    background_tasks.add_task(session_logger.remove)
    return DocumentsSimilarityOutputSchema()
