from abc import abstractmethod
from src.misc.document_schema import DocumentStructureSchema


class BaseDB:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def connect(self, **kwargs):
        raise NotImplementedError(
            "connection is not implemented in base database class"
        )

    @abstractmethod
    def push_document(self, document_structured_extraction: DocumentStructureSchema):
        raise NotImplementedError(
            "connection is not implemented in base database class"
        )
