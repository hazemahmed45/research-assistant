"""
API Settings Module
=====================

Defines settings classes for the API, including base settings and model-specific settings.

Classes
-------
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """
    Base settings for the API, including authentication, logging, and deployment configuration, settings gets read from .env file


    :param logdir: Path to the local logs directory (default: ``logs``)
    :type logdir: str
    :param api_host: API host (default: ``0.0.0.0``)
    :type api_host: str
    :param api_port: API port (default: ``8000``)
    :type api_port: int
    :param main_route: Main API route (default: ``api``)
    :type main_route: str
    :param cache_dir: Cache directory (default: ``None``)
    :type cache_dir: bool


    **Settings Configuration**

    * **env_file**: Environment file (default: ``.env``)
    * **env_ignore_empty**: Ignore empty environment variables (default: ``True``)
    * **env_file_encoding**: Environment file encoding (default: ``utf-8``)
    * **extra**: Extra settings configuration (default: ``ignore``)
    """

    logdir: str = Field(default="logs", description="path of the logs directory")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    main_route: str = "api"
    cache_folder: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
    )


if __name__ == "__main__":
    print(ApiSettings().model_dump_json())
