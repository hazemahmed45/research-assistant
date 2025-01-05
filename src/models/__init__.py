"""Research Assistant Model Factory

This module provides a factory for creating various research assistant models.
"""

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Union
from pydantic_settings import BaseSettings
from src.enums import (
    EmbeddingModelCards,
    EmbeddingModelRunnerTypes,
    LLMModelCards,
    LLMServingType,
    VectorstoreType,
)
from src.models.doc_analyizer import (
    BaseDocumentAnalyizer,
    DocumentAnalyizerHuggingFaceAPILLM,
    DocumentAnalyizerPipelineLLM,
    DocumentAnalyizerOpenAIAPILLM,
    DocumentAnalyizerSettings,
)
from src.models.doc_embedder import (
    BaseDocumentEmbedder,
    CohereDocumentEmbedder,
    OpenaiDocumentEmbedder,
    HuggingfaceDocumentEmbedder,
    DocumentEmbedderSettings,
)
from src.models.doc_vectorstore import (
    BaseVectorstore,
    ChromaVectorstore,
    FaissVectorStore,
    VectorstoreSettings,
)

from src.misc.exceptions import (
    ModelTypeNotSupported,
    ModelServingTypeNotSupported,
    VectorstoreTypeNotSupported,
)


class ModelFactory:
    """Research Assistant Model Factory

    This class provides a factory for creating data analyizer.
    """

    class FactoryUnpackValues(Enum):
        DOCUMENT_ANALYIZER = "document_analyizer"
        DOCUMENT_EMBEDDER = "document_embedder"
        VECTORSTORE = "vectorstore"

    @staticmethod
    def build_model(
        document_analyizer_settings: DocumentAnalyizerSettings = None,
        document_embedder_settings: DocumentEmbedderSettings = None,
        vectorstore_settings: VectorstoreSettings = None,
        cache_dir: str = None,
    ) -> Dict[str, Union[BaseDocumentAnalyizer, BaseDocumentEmbedder, BaseVectorstore]]:
        """Model factory

        :param document_analyizer_settings: Document analyizer model configuration settings
        :type document_analyizer_settings: DocumentAnalyizerSettings
        :param document_embedder_settings: Document embedder model configuration settings
        :type document_embedder_settings: DocumentEmbedderSettings
        :param vectorstore_settings: Vectorstore configuration settings
        :type vectorstore_settings: VectorstoreSettings
        :param authentication_token: Authentication token for model APIs
        :type authentication_token: str
        :param cache_dir: Directory for caching model downloads (optional)
        :type cache_dir: str
        :return: The built results, if any of the settings is passed to model factory, will be included in the build dictionery
        :rtype:  Dict[str,Union[BaseDocumentAnalyizer,BaseDocumentEmbedder,BaseVectorstore]]
        :raises ModelTypeNotSupported: If the model type is not supported
        :raises ModelServingTypeNotSupported: If the model runner is not supported
        :raises VectorstoreTypeNotSupported: If the vectorstore is not supported
        """
        built_results = defaultdict(None)
        if document_analyizer_settings is not None:
            built_results[ModelFactory.FactoryUnpackValues.DOCUMENT_ANALYIZER] = (
                ModelFactory._get_document_analyizer(
                    settings=document_analyizer_settings,
                )
            )
        if document_embedder_settings is not None:
            built_results[ModelFactory.FactoryUnpackValues.DOCUMENT_EMBEDDER] = (
                ModelFactory._get_document_embedder(
                    settings=document_embedder_settings,
                    cache_dir=cache_dir,
                )
            )
        if vectorstore_settings is not None:
            if (
                built_results[ModelFactory.FactoryUnpackValues.DOCUMENT_EMBEDDER]
                == None
            ):
                raise ValueError(
                    "should pass the document embedder settings with vectorstore settings to the factory at the same time"
                )
            built_results[ModelFactory.FactoryUnpackValues.VECTORSTORE] = (
                ModelFactory._get_vectorstore(
                    settings=vectorstore_settings,
                    document_embedder=built_results[
                        ModelFactory.FactoryUnpackValues.DOCUMENT_EMBEDDER
                    ],
                )
            )
        return built_results

    @staticmethod
    def _get_document_analyizer(
        settings: DocumentAnalyizerSettings,
    ) -> BaseDocumentAnalyizer:
        """Retrieve a data analyizer API Model

        :param settings: data analyizer model configuration settings
        :type settings: DocumentAnalyizerSettings
        :param authentication_token: Authentication token for model APIs
        :type authentication_token: str
        :return: The data analyizer API model
        :rtype: Union[ DocumentAnalyizerHuggingFaceAPILLM, DocumentAnalyizerVLLMOpenAIAPILLM ]
        :raises ModelServingTypeNotSupported: If the serving type is not supported
        :raises ModelTypeNotSupported: If the model type is not supported
        """
        if not any([s_type == settings.llm_serving_type for s_type in LLMServingType]):
            raise ModelServingTypeNotSupported(
                f"{settings.llm_serving_type.value} is not supported in this repo"
            )
        if not any([s_type == settings.llm_model_card for s_type in LLMModelCards]):
            raise ModelTypeNotSupported(
                f"{settings.llm_model_card.value} is not supported in this repo"
            )
        if settings.llm_serving_type == LLMServingType.OPENAI:
            return DocumentAnalyizerOpenAIAPILLM(
                llm_api_url=settings.llm_api_url,
                auth_token=settings.llm_auth_token,
                model_name=settings.llm_model_card.value,
                max_new_tokens=settings.max_new_tokens,
                top_p=settings.top_p,
                temperature=settings.temperature,
                num_return_sequences=settings.num_return_sequences,
                max_time=settings.max_time,
                prompt_preprocessing=settings.prompt_preprocessing,
            )
        elif settings.llm_serving_type == LLMServingType.HUGGINGFACE_API:
            return DocumentAnalyizerHuggingFaceAPILLM(
                llm_api_url=settings.llm_api_url,
                auth_token=settings.llm_auth_token,
                model_name=settings.llm_model_card.value,
                max_new_tokens=settings.max_new_tokens,
                top_k=settings.top_k,
                top_p=settings.top_p,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
                num_return_sequences=settings.num_return_sequences,
                max_time=settings.max_time,
                prompt_preprocessing=settings.prompt_preprocessing,
            )
        elif settings.llm_serving_type == LLMServingType.HUGGINGFACE:
            return DocumentAnalyizerPipelineLLM(
                model_name=settings.llm_model_card.value,
                prompt_preprocessing=settings.prompt_preprocessing,
                auth_token=settings.llm_auth_token,
                return_full_text=False,
                use_cache=True,
                max_new_tokens=settings.max_new_tokens,
                top_k=settings.top_k,
                top_p=settings.top_p,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
                num_return_sequences=settings.num_return_sequences,
                max_time=settings.max_time,
            )
        else:
            raise ModelServingTypeNotSupported(
                f"{settings.llm_serving_type.value} is not supported in this repo"
            )

    @staticmethod
    def _get_document_embedder(
        settings: DocumentEmbedderSettings, cache_dir: str = None
    ) -> BaseDocumentEmbedder:
        """Retrieve a vectorstore

        :param settings: vectorstore model configuration settings
        :type settings: VectorstoreSettings
        :return: the vectorstore
        :rtype: Union[HuggingfaceDocumentEmbedder,OpenaiDocumentEmbedder,CohereDocumentEmbedder]
        :raises ModelTypeNotSupported: If the model type is not supported
        :raises ModelServingTypeNotSupported: If the model runner is not supported
        """
        if not any(
            [s_type == settings.embedding_model_card for s_type in EmbeddingModelCards]
        ):
            raise ModelTypeNotSupported(
                f"{settings.embedding_model_card.value} is not supported in this repo"
            )
        if not any(
            [
                s_type == settings.embedding_runner_type
                for s_type in EmbeddingModelRunnerTypes
            ]
        ):
            raise ModelServingTypeNotSupported(
                f"{settings.embedding_runner_type.value} is not supported in this repo"
            )
        if settings.embedding_runner_type == EmbeddingModelRunnerTypes.HUGGINGFACE:
            return HuggingfaceDocumentEmbedder(
                embedding_model_card=settings.embedding_model_card.value,
                auth_token=settings.embedding_auth_token,
                cache_dir=cache_dir,
                device=settings.device,
            )
        elif settings.embedding_runner_type == EmbeddingModelRunnerTypes.OPENAI:
            return OpenaiDocumentEmbedder(
                embedding_model_card=settings.embedding_model_card.value,
                auth_token=settings.embedding_auth_token,
            )
        elif settings.embedding_runner_type == EmbeddingModelRunnerTypes.COHERE:
            return CohereDocumentEmbedder(
                embedding_model_card=settings.embedding_model_card.value,
                auth_token=settings.embedding_auth_token,
            )
        else:
            raise ModelTypeNotSupported(
                f"{settings.embedding_model_card.value} is not supported in this repo yet"
            )

    @staticmethod
    def _get_vectorstore(
        settings: VectorstoreSettings,
        document_embedder: BaseDocumentEmbedder,
    ) -> BaseVectorstore:
        """Retrieve a vectorstore

        :param settings: vectorstore model configuration settings
        :type settings: VectorstoreSettings
        :return: the vectorstore
        :rtype: Union[ChromaVectorstore,FaissVectorStore]
        :raises VectorstoreTypeNotSupported: If the vectorstore is not supported
        """
        if not any([s_type == settings.vectorstore_type for s_type in VectorstoreType]):
            raise VectorstoreTypeNotSupported(
                f"{settings.vectorstore_type.value} is not supported in this repo"
            )
        if settings.vectorstore_type == VectorstoreType.CHROME:
            return ChromaVectorstore(
                collection_name=settings.vectorstore_name,
                persist_directory=settings.persist_directory,
                document_embedder=document_embedder,
            )
        elif settings.vectorstore_type == VectorstoreType.FAISS:
            return FaissVectorStore(
                vectorstore_name=settings.vectorstore_name,
                persist_directory=settings.persist_directory,
                document_embedder=document_embedder,
            )

        else:
            raise VectorstoreTypeNotSupported(
                f"{settings.vectorstore_type.value} vectorstore is not supported yet"
            )


class SingletonModelFactory(ModelFactory):
    built_results = defaultdict(None)

    @staticmethod
    def _get_document_analyizer(settings, new_model=False) -> BaseDocumentAnalyizer:
        if (
            SingletonModelFactory.built_results[
                SingletonModelFactory.FactoryUnpackValues.DOCUMENT_ANALYIZER
            ]
            == None
            or new_model
        ):
            SingletonModelFactory.built_results[
                SingletonModelFactory.FactoryUnpackValues.DOCUMENT_ANALYIZER
            ] = super()._get_document_analyizer(settings)
        return SingletonModelFactory.built_results[
            SingletonModelFactory.FactoryUnpackValues.DOCUMENT_ANALYIZER
        ]

    @staticmethod
    def _get_document_embedder(
        settings, cache_dir=None, new_model=False
    ) -> BaseDocumentEmbedder:
        if (
            SingletonModelFactory.built_results[
                SingletonModelFactory.FactoryUnpackValues.DOCUMENT_EMBEDDER
            ]
            == None
            or new_model
        ):
            SingletonModelFactory.built_results[
                SingletonModelFactory.FactoryUnpackValues.DOCUMENT_EMBEDDER
            ] = super()._get_document_embedder(settings, cache_dir)
        return SingletonModelFactory.built_results[
            SingletonModelFactory.FactoryUnpackValues.DOCUMENT_EMBEDDER
        ]

    @staticmethod
    def _get_vectorstore(
        settings: VectorstoreSettings,
        document_embedder: BaseDocumentEmbedder,
        new_vectorstore=False,
    ) -> BaseVectorstore:
        if (
            SingletonModelFactory.built_results[
                SingletonModelFactory.FactoryUnpackValues.VECTORSTORE
            ]
            == None
            or new_vectorstore
        ):
            SingletonModelFactory.built_results[
                SingletonModelFactory.FactoryUnpackValues.VECTORSTORE
            ] = super()._get_vectorstore(settings, document_embedder)
        return SingletonModelFactory.built_results[
            SingletonModelFactory.FactoryUnpackValues.VECTORSTORE
        ]


if __name__ == "__main__":
    pass
