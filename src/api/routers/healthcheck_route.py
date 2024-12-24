"""
Healthcheck API Module

This module defines the API endpoint for healthcheck.
It provides a simple endpoint to verify the API's availability.

"""

from __future__ import annotations
import sys
from fastapi import APIRouter, status, BackgroundTasks
from fastapi.responses import JSONResponse
import loguru
from loguru import logger

from src.api.config import ApiSettings
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id

REQUEST_TYPE = "healthcheck"

settings = ApiSettings()
healthcheck_router = APIRouter()


@healthcheck_router.get("/healthcheck", tags=["Healthcheck"])
async def healthcheck(
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    **Verify API Availability Route**

    Simple healthcheck endpoint to confirm API responsiveness

    :param background_tasks: Handler for scheduling background tasks
    :type background_tasks: BackgroundTasks
    :return: Success response with a status message
    :rtype: JSONResponse

    **Logging Behavior**

    * Generates a unique request ID for logging purposes
    * Configures logging to:
        + Standard Error (stderr)
        + Local log file via `FileHandler`
    * Logs the receipt of the request with the unique ID
    * Schedules log sink removal as a background task upon response completion

    **Response Details**

    * **JSON Response Body**: ``{"message": "I am alive"}``
    * **HTTP Status Code**: 200 OK
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
    background_tasks.add_task(session_logger.remove)
    return JSONResponse({"message": "I am alive"}, status_code=status.HTTP_200_OK)
