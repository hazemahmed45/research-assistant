from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.database.base_db import BaseDB
from src.database.mongodb import MongoDB
from src.enums import DatabaseTypes
from src.misc.exceptions import DatabaseTypeNotSupported


class DBSettings(BaseSettings):
    db_type: DatabaseTypes = Field(default=DatabaseTypes.MONGODB)
    db_client: str = Field(default="localhost:27017")
    db_name: str = Field(default="research-papers-db")
    db_username: Optional[str] = Field(default=None)
    db_password: Optional[str] = Field(default=None)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


class DatabaseFactory:
    @staticmethod
    def initialize_database(db_settings: DBSettings) -> BaseDB:
        if not any([db_type == db_settings.db_type for db_type in DatabaseTypes]):
            raise DatabaseTypeNotSupported(
                f"{db_settings.db_type.value} is not supported in this repo"
            )
        if db_settings.db_type == DatabaseTypes.MONGODB:
            return MongoDB(
                mongodb_url=db_settings.db_client,
                database_name=db_settings.db_name,
                username=db_settings.db_username,
                password=db_settings.db_password,
            )
        elif db_settings.db_type == DatabaseTypes.NOTION:
            return MongoDB(
                mongodb_url=db_settings.db_client,
                database_name=db_settings.db_name,
                username=db_settings.db_username,
                password=db_settings.db_password,
            )
        else:
            raise DatabaseTypeNotSupported(
                f"{db_settings.db_type.value} is not supported in this repo"
            )
        return

    @staticmethod
    def build(db_settings: DBSettings) -> BaseDB:
        return DatabaseFactory.initialize_database(db_settings=db_settings)


class SingletonDatabaseFactory(DatabaseFactory):
    db_instance: BaseDB | None = None

    @staticmethod
    def build(db_settings: DBSettings) -> BaseDB:
        if SingletonDatabaseFactory.db_instance is None:
            SingletonDatabaseFactory.db_instance = (
                SingletonDatabaseFactory.initialize_database(db_settings)
            )
        return SingletonDatabaseFactory.db_instance
