from enum import Enum
import pymongo
from pymongo import MongoClient
from src.database.base_db import BaseDB
from src.misc.document_schema import DocumentStructureSchema


class MongoDB(BaseDB):
    class DatabaseNames(Enum):
        DOCUMENT_SUMMARIZATION_DATABASE = "documents-summarization"

    def __init__(
        self,
        mongodb_host: str = "localhost",
        mongodb_port: str = "27017",
        database_name: str = "research-papers-db",
        username: str = None,
        password: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mongodb_host = mongodb_host
        self.mongodb_port = mongodb_port
        self.username = username
        self.password = password
        self.database_name = database_name
        self.client = self.connect()

    def connect(self, **kwargs) -> MongoClient:
        connection_uri = f"mongodb://{self.username+':'+self.password if self.username is not None else ''}@{self.mongodb_host}:{self.mongodb_port}/"
        return MongoClient(connection_uri)

    def push_document(self, document_structured_extraction: DocumentStructureSchema):
        document_summarization_db = self.client[
            self.DatabaseNames.DOCUMENT_SUMMARIZATION_DATABASE.value
        ]

        return super().push_document(document_structured_extraction)
