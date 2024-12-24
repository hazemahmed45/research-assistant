from src.api.app import create_app

# from loguru import logger
import uvicorn
from fastapi import FastAPI
from uvicorn.config import LOG_LEVELS
from src.api.config import ApiSettings

app: FastAPI = create_app()

settings = ApiSettings()


def main():
    # logger.info("API STARTING")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        loop="auto",
        log_level="info",
    )
    # logger.info("API STOPPING")


if __name__ == "__main__":
    main()
