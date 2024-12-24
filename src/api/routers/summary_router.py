"""
Summarize Document API Module

This module defines the API endpoint for summarize document.
It provides an endpoint to summarize the document given in the request body

"""

from __future__ import annotations
import sys
from fastapi import APIRouter, BackgroundTasks
import loguru
from loguru import logger

from src.api.config import ApiSettings
from src.api.schema import (
    DocumentSummaryInputSchema,
    DocumentSummaryOutputSchema,
)
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id

REQUEST_TYPE = "sum-doc"

settings = ApiSettings()
sum_doc_router = APIRouter()


@sum_doc_router.get("/sum_doc", tags=["Summarize Document"])
async def similar_doc(
    schema: DocumentSummaryInputSchema,
    background_tasks: BackgroundTasks,
) -> DocumentSummaryOutputSchema:
    """
    TODO docstring
    """
    request_unique_id: str = create_unique_user_id()
    session_logger: loguru.Logger = logger.bind(
        user_unique_id=request_unique_id, request_type=REQUEST_TYPE
    )
    session_logger.remove()
    session_logger.add(
        sys.stderr,
        format="<g>{time}</g> | <m>{level}</m> | <e>{name}:{function}:{line}</e> | REQUEST ID -> {extra[user_unique_id]} : {message}",
    )

    session_logger.add(
        sink=FileHandler(user_unique_id=request_unique_id),
        format="<g>{time}</g> | <m>{level}</m> | <e>{name}:{function}:{line}</e> | REQUEST ID -> {extra[user_unique_id]} : {message}",
    )

    session_logger.info("Request Recieved")

    # TODO IMPLEMENT SUMMARIZE DOCUMENTS ROUTE
    background_tasks.add_task(session_logger.remove)
    return DocumentSummaryOutputSchema()
