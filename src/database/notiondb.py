from enum import Enum
import notion
from notion.client import NotionClient
from src.database.base_db import BaseDB
from src.misc.document_schema import DocumentStructureSchema


class NotionDB(BaseDB):
    class DatabaseNames(Enum):
        DOCUMENT_SUMMARIZATION_DATABASE = "documents-summarization"

    def __init__(
        self,
        token: str,
        database_name: str = "research-papers-db",
        email: str = None,
        password: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token = token
        self.email = email
        self.password = password
        self.database_name = database_name
        self.client: NotionClient = self.connect()

    def connect(self, **kwargs) -> NotionClient:
        return NotionClient(
            token_v2=self.token, email=self.email, password=self.password
        )

    # def push_document(self, document_structured_extraction: DocumentStructureSchema):

    #     return super().push_document(document_structured_extraction)
