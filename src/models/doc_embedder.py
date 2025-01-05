from abc import abstractmethod
from typing import List, Optional, Union, Literal
import os

from dotenv import get_key as get_dotenv_key
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from langchain_core.embeddings import Embeddings

os.environ["HUGGINGFACEHUB_API_TOKEN"] = get_dotenv_key(".env", "EMBEDDING_AUTH_TOKEN")
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

os.environ["OPENAI_API_KEY"] = get_dotenv_key(".env", "EMBEDDING_AUTH_TOKEN")
from langchain_openai.embeddings import OpenAIEmbeddings

os.environ["COHERE_API_KEY"] = get_dotenv_key(".env", "EMBEDDING_AUTH_TOKEN")
from langchain_cohere.embeddings import CohereEmbeddings
from src.enums import EmbeddingModelCards, EmbeddingModelRunnerTypes


class DocumentEmbedderSettings(BaseSettings):
    """
    **Configuration settings for the Document Embedder **

    :param document_embedder_modelcard: embedding model card
    :type document_embedder_modelcard: EmbeddingModelCards
    :param document_embedder_runner_type: the type of runner to run the embedding model (not all models can run on all runners)
    :type document_embedder_runner_type: EmbeddingModelRunnerTypes
    :param device: the device to run the embedding model, applicable if the runner is local
    :type device: 'cuda' | 'cpu' | None

    """

    embedding_model_card: EmbeddingModelCards = Field(
        default=EmbeddingModelCards.MODERNBERT, description="embedding model card"
    )
    embedding_runner_type: EmbeddingModelRunnerTypes = Field(
        default=EmbeddingModelRunnerTypes.HUGGINGFACE,
        description="the type of runner to run the embedding model (not all models can run on all runners)",
    )
    embedding_auth_token: str = Field(default=None)
    device: Optional[Literal["cuda", "cpu"]] = Field(
        default="cpu",
        description="the device to run the embedding model, applicable if the runner is local",
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


class BaseDocumentEmbedder:
    def __init__(self, embedding_model_card: str, auth_token: str):
        self._embedding_model_card = embedding_model_card
        self.authentication_token = auth_token
        self._embedding_model_args = self.set_args()
        self.embedding_model: Embeddings = self.build_embedding_model()

    def embed_documents(self, documents: Union[List[str], str]) -> List[List[float]]:
        assert any(
            [isinstance(documents, instance_type) for instance_type in [str, list]]
        ), "document needs to be either string or a list of string"
        if isinstance(documents, str):
            documents = [documents]

        return self.embedding_model.embed_documents(documents)

    @abstractmethod
    def build_embedding_model(self) -> Embeddings:
        raise NotImplementedError(
            "Build embedding model function is not yet implemented"
        )

    @abstractmethod
    def set_args(self):
        raise NotImplementedError("Set argumnet function is not yet implemented")


class HuggingfaceDocumentEmbedder(BaseDocumentEmbedder):
    def __init__(
        self,
        embedding_model_card: str,
        auth_token: str,
        cache_dir: str,
        device: str = "cpu",
    ):
        self.cache_dir = cache_dir
        self.device = device
        super().__init__(embedding_model_card, auth_token=auth_token)

    def build_embedding_model(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self._embedding_model_card, **self._embedding_model_args
        )

    def set_args(self):
        return {
            "model_kwargs": {
                "device": self.device,
                "token": self.authentication_token,
            },
            "encode_kwargs": {"normalize_embeddings": False},
        }


class CohereDocumentEmbedder(BaseDocumentEmbedder):
    def __init__(
        self,
        embedding_model_card: str,
        auth_token: str,
    ):
        super().__init__(embedding_model_card, auth_token=auth_token)

    def build_embedding_model(self) -> CohereEmbeddings:
        return CohereEmbeddings(
            model=self._embedding_model_card, **self._embedding_model_args
        )

    def set_args(self):
        return {
            "cohere_api_key": self.authentication_token,
            "request_timeout": 1000,
            "max_retries": 3,
        }


class OpenaiDocumentEmbedder(BaseDocumentEmbedder):
    def __init__(
        self,
        embedding_model_card,
        auth_token: str,
    ):
        super().__init__(embedding_model_card, auth_token=auth_token)

    def build_embedding_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self._embedding_model_card, **self._embedding_model_args
        )

    def set_args(self):
        return {
            "api_key": self.authentication_token,
            "timeout": 1000,
            "max_retries": 3,
        }
