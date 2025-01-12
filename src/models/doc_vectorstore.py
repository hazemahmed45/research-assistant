from abc import abstractmethod
from enum import Enum
from json import load
from re import S
from typing import Any, Dict, List, Union, Optional
import os
import shutil
from uuid import uuid4
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from langchain_chroma.vectorstores import Chroma
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

from src.misc.document_schema import DocumentStructureSchema
from src.enums import VectorstoreType
from src.models.doc_embedder import BaseDocumentEmbedder
from src.misc.create_unique_id import create_unique_id_from_str


class VectorstoreSettings(BaseSettings):
    """
    **Configuration settings for the Vectorstore**

    :param vectorstore_type: the type of vectorstore to use
    :type vectorstore_type: VectorstoreType
    :param vectorstore_name: the name of the vectorstore collection
    :type vectorstore_name: str
    :param persist_directory: Where to save data locally, remove if not necessary
    :type persist_directory: str
    :param document_distance_threshold: threshold for filtering not so similar documents.
    :type document_distance_threshold: float

    """

    vectorstore_type: VectorstoreType = Field(
        default=VectorstoreType.CHROME, description="the type of vectorstore to use"
    )
    vectorstore_name: str = Field(
        default="document-embedding-store",
        description="the name of the vectorstore collection",
    )
    persist_directory: str = Field(
        default="document_vectorstore_db",
        description="Where to save data locally, remove if not necessary",
    )
    document_distance_threshold: float = Field(
        default=0.3, description="threshold for filtering not so similar documents"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


class BaseVectorstore:
    class FilterMetadataKeys(Enum):
        SOURCE = "source"
        TAG = "tag"

    class FilterDocumentStructureTags(Enum):
        MOTIVIATION = "motivation"
        PROBLEMS = "problems"
        CHALLENGES = "challenges"
        CONTRIBUTION = "contribution"
        TECHNIQUES = "techniques"
        METHODOLOGY = "methodology"
        PROPOSED_MODEL = "proposed_model"
        RESULTS = "results"
        FUTURE_WORK = "future_work"

    def __init__(
        self,
        vectorstore_name: str = "document-embedding-store",
        persist_directory: str = "document_vectorstore_db",
        document_embedder: BaseDocumentEmbedder = None,
        distance_threshold: float = 0.3,
    ):
        self.vectorstore_name = vectorstore_name
        self.persist_directory = persist_directory
        self._document_embedder: BaseDocumentEmbedder = document_embedder
        self._vector_store: VectorStore = self.set_vectorstore()
        self.distance_threshold = distance_threshold

    def retrieve_similar_docs(
        self,
        document: Document,
        topk=10,
        metadata: Dict[str, Any] = None,
    ) -> List[Document]:
        if metadata is None:
            metadata = {}
        retrieved_results: List[tuple[Document, float]] = (
            self._vector_store.similarity_search(
                query=document.page_content, k=topk, filter=metadata
            )
        )
        return [
            similar_doc
            for similar_doc in retrieved_results
            # if score < self.distance_threshold and score != 0.0
        ]

    def get_docs_by_ids(self, doc_ids: Union[str, List[str]]) -> List[Document]:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        return self._vector_store.get_by_ids(doc_ids)

    def add_document(
        self,
        documents: Union[List[Document], Document],
    ) -> List[str]:
        if isinstance(documents, Document):
            documents = [documents]

        doc_ids = [d.id for d in documents]

        found_doc_ids = [doc.id for doc in self.get_docs_by_ids(doc_ids)]

        filtered_documents = [doc for doc in documents if doc.id not in found_doc_ids]
        filtered_doc_ids = [doc.id for doc in filtered_documents]
        if len(filtered_documents) > 0:
            self._vector_store.add_documents(
                documents=filtered_documents, ids=filtered_doc_ids
            )
        return filtered_doc_ids

    def add_structured_document(self, structured_document: DocumentStructureSchema):
        structured_docs: List[Document] = [
            Document(
                page_content=structured_document.model_dump()[tag.value],
                metadata={
                    self.FilterMetadataKeys.SOURCE.value: str(structured_document.link),
                    self.FilterMetadataKeys.TAG.value: tag.value,
                },
                id=create_unique_id_from_str(
                    str(structured_document.link) + f"_{tag.value}"
                ),
            )
            for tag in self.FilterDocumentStructureTags
        ]
        structured_document_ids = {
            doc_id: doc.metadata[self.FilterMetadataKeys.TAG.value]
            for doc_id, doc in zip(
                self.add_document(documents=structured_docs), structured_docs
            )
        }

        return structured_document_ids

    def get_document_embedder(self) -> BaseDocumentEmbedder:
        return self._document_embedder

    @abstractmethod
    def set_vectorstore(self):
        raise NotImplementedError("set vector store is not implemented in base class")

    @abstractmethod
    def save_vectorstore(self):
        raise NotImplementedError("save vector store is not implemented for this class")

    @abstractmethod
    def load_vectorstore(self):
        raise NotImplementedError("load vector store is not implemented for this class")


class ChromaVectorstore(BaseVectorstore):
    def __init__(
        self,
        collection_name: str = "document-embedding-store",
        persist_directory: str = "document_vectorstore_db",
        document_embedder: BaseDocumentEmbedder = None,
    ):
        super().__init__(
            vectorstore_name=collection_name,
            persist_directory=persist_directory,
            document_embedder=document_embedder,
        )
        self._vector_store: Chroma

    def set_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.vectorstore_name,
            embedding_function=self.get_document_embedder(),
            persist_directory=self.persist_directory,
        )

    def get_docs_by_ids(self, doc_ids: Union[List[str], str]) -> List[Document]:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        doc_list: List[Document] = []
        retrieved_dict = self._vector_store.get(ids=doc_ids)
        for doc_id, doc_content, doc_metadata in zip(
            retrieved_dict["ids"],
            retrieved_dict["documents"],
            retrieved_dict["metadatas"],
        ):
            doc_list.append(
                Document(page_content=doc_content, metadata=doc_metadata, id=doc_id)
            )
        return doc_list

    def save_vectorstore(self):
        pass
        return

    def load_vectorstore(self):
        pass
        return


class FaissVectorStore(BaseVectorstore):
    def __init__(
        self,
        vectorstore_name: str = "document-embedding-store",
        persist_directory: str = "document_vectorstore_db",
        document_embedder: BaseDocumentEmbedder = None,
    ):
        super().__init__(
            vectorstore_name=vectorstore_name,
            persist_directory=persist_directory,
            document_embedder=document_embedder,
        )
        self._vector_store: FAISS

    def set_vectorstore(self) -> FAISS:
        import faiss

        return FAISS(
            index=faiss.IndexFlatL2(
                len(self.get_document_embedder().embed_documents("hello world")[0])
            ),
            embedding_function=self.get_document_embedder().embedding_model,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=True,
            distance_strategy=DistanceStrategy.COSINE,
        )

    def save_vectorstore(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        os.makedirs(self.persist_directory)
        self._vector_store.save_local(
            folder_path=self.persist_directory, index_name=self.vectorstore_name
        )
        return

    def load_vectorstore(self):
        if os.path.exists(os.path.join(self.persist_directory, self.vectorstore_name)):
            self._vector_store = FAISS.load_local(
                folder_path=self.persist_directory,
                embeddings=self._document_embedder.embedding_model,
                index_name=self.vectorstore_name,
            )
        return


if __name__ == "__main__":
    # print(list(BaseVectorstore.FilterDocumentStructureTags))
    pass
