"""
API Creation Module
=====================

Provides functions for creating the FastAPI application instance.

Functions
---------
"""

from pathlib import Path
from fastapi import FastAPI, APIRouter
from src.api.config import ApiSettings
from src.api.routers.healthcheck_route import healthcheck_router
from src.api.routers.compare_router import comp_doc_router
from src.api.routers.similar_router import sim_doc_router
from src.api.routers.summary_router import sum_doc_router


def get_active_branch_name():
    """
    Retrieve the name of the currently active Git branch

    :return: Name of the active Git branch
    :rtype: str
    """
    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def create_api_desc() -> str:
    """
    Generate a brief description for the API

    :return: API description string
    :rtype: str
    """
    desc = f"""Deployed API for research assistant"""
    return desc


def create_app() -> FastAPI:
    """
    Initialize and configure the FastAPI application instance

    **Application Configuration**

    * **Title**: Voice Assistant API
    * **Summary**: Brief API description (generated by `create_api_desc`)
    * **Version**: Currently active Git branch name (retrieved by `get_active_branch_name`)
    * **Contact**: API maintainer information (name and email)
    * **Docs/Redoc URLs**: Custom URLs for API documentation and Redoc

    **Router Inclusion**

    * **Healthcheck Router**: Always included
    * **Summary Doc Router**: Always included
    * **Compare Doc Router**: Always included
    * **Similar Doc Router**: Always included

    :return: Configured FastAPI application instance
    :rtype: FastAPI

    """
    settings = ApiSettings()
    api_desc = create_api_desc()
    app: FastAPI = FastAPI(
        title="Research Assistant API",
        summary=api_desc,
        # description=create_api_desc(),
        version=get_active_branch_name(),
        docs_url=f"/{settings.main_route}/docs",
        redoc_url=f"/{settings.main_route}/redoc",
    )
    api_router = APIRouter(prefix=f"/{settings.main_route}")
    api_router.include_router(healthcheck_router)
    api_router.include_router(sum_doc_router)
    api_router.include_router(comp_doc_router)
    api_router.include_router(sim_doc_router)

    app.include_router(api_router)

    return app
