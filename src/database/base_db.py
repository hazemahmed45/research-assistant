from abc import abstractmethod
from typing import List
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
    def push_document(self, structured_document: DocumentStructureSchema) -> None:
        raise NotImplementedError(
            "push document is not implemented in base database class"
        )

    @abstractmethod
    def push_documents(
        self, structured_documents: List[DocumentStructureSchema]
    ) -> None:
        raise NotImplementedError(
            "push documents is not implemented in base database class"
        )

    @abstractmethod
    def retrieve_structure_document(
        self, document_link: str
    ) -> DocumentStructureSchema:
        raise NotImplementedError(
            "retrieve document is not implemented in base database class"
        )

    @abstractmethod
    def retrieve_structure_documents(
        self, documents_link: List[str]
    ) -> List[DocumentStructureSchema]:
        raise NotImplementedError(
            "retrieve documents is not implemented in base database class"
        )
