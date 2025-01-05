from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from src.enums import DatabaseTypes


class DBSettings(BaseSettings):
    db_type: DatabaseTypes = Field(default=DatabaseTypes.MONGODB)
    db_host: str = Field(default="localhost")
    db_port: str = Field(default="27017")
    db_name: str = Field(default="research-papers-db")
    db_username: Optional[str] = Field(default=None)
    db_password: Optional[str] = Field(default=None)
