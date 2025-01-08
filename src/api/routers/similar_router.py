"""
Similar Document API Module

This module defines the API endpoint for similar document.
It provides an endpoint to retrieve similar documents to the given document in the request body

"""

from __future__ import annotations
import sys
from typing import Union, List
from fastapi import APIRouter, BackgroundTasks
import loguru
from loguru import logger
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents.base import Document
from src.api.config import ApiSettings
from src.models import (
    SingletonModelFactory,
    DocumentAnalyizerSettings,
    DocumentEmbedderSettings,
    VectorstoreSettings,
    BaseDocumentEmbedder,
    BaseDocumentAnalyizer,
    BaseVectorstore,
)
from src.database import SingletonDatabaseFactory, DBSettings, BaseDB
from src.misc.document_schema import DocumentStructureSchema

from src.api.schema import (
    DocumentsSimilarityOutputSchema,
    DocumentsSimilarityInputSchema,
)
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id
from src.misc.utils import merge_pages_in_document
from src.enums import Constants

# REQUEST_TYPE = "sim-doc"

api_settings = ApiSettings()
sim_doc_router = APIRouter()

factory_results: loguru.Dict[
    str, BaseDocumentAnalyizer | BaseDocumentEmbedder | BaseVectorstore
] = SingletonModelFactory.build_model(
    document_analyizer_settings=DocumentAnalyizerSettings(),
    document_embedder_settings=DocumentEmbedderSettings(),
    vectorstore_settings=VectorstoreSettings(),
    cache_dir=api_settings.cache_folder,
)
document_analyizer: BaseDocumentAnalyizer = factory_results[
    SingletonModelFactory.FactoryUnpackValues.DOCUMENT_ANALYIZER
]
vectorstore: BaseVectorstore = factory_results[
    SingletonModelFactory.FactoryUnpackValues.VECTORSTORE
]

database: BaseDB = SingletonDatabaseFactory.build(db_settings=DBSettings())


@sim_doc_router.post("/sim_doc", tags=["Similar Document"])
async def similar_doc(
    schema: DocumentsSimilarityInputSchema,
    background_tasks: BackgroundTasks,
) -> DocumentsSimilarityOutputSchema:
    """
    TODO docstring
    """
    request_unique_id: str = create_unique_user_id()
    session_logger: loguru.Logger = logger.bind(
        user_unique_id=request_unique_id,
        request_type=Constants.SIMILAR_DOCUMENT_REQUEST_TYPE.value,
    )
    session_logger.remove()
    session_logger.add(
        sys.stderr,
        format=Constants.LOGGER_REQUEST_FORMAT.value,
    )

    session_logger.add(
        sink=FileHandler(user_unique_id=request_unique_id),
        format=Constants.LOGGER_REQUEST_FORMAT.value,
    )

    session_logger.info("Request Recieved")

    # TODO IMPLEMENT SIMILAR DOCUMENTS ROUTE
    structured_document: Union[DocumentStructureSchema, None] = (
        database.get_structured_document_by_link(document_link=str(schema.link))
    )
    if structured_document == None:
        pdf_loader = PyPDFLoader(file_path=schema.link)
        document_pages: List[Document] = pdf_loader.load()
        session_logger.info(f"Loaded {len(document_pages)} pages")
        document: Document = merge_pages_in_document(document_pages=document_pages)

        session_logger.info(f"Merged all pages into one document with id {document.id}")

        session_logger.info("Start document analyizing")
        structured_document: DocumentStructureSchema = document_analyizer(
            doc_context=document
        )
        session_logger.info("Finished document analyizing")
        background_tasks.add_task(
            vectorstore.add_structured_document,
            structured_document=structured_document,
        )
        background_tasks.add_task(
            database.push_document, structured_document=structured_document
        )
    retrieved_docs = vectorstore.retrieve_similar_docs(
        document=Document(
            page_content=structured_document.model_dump()[
                schema.similarity_category.value
            ]
        ),
        topk=schema.topk,
        metadata={
            BaseVectorstore.FilterMetadataKeys.TAG.value: schema.similarity_category.value
        },
    )
    # print(structured_document)
    print(retrieved_docs)
    background_tasks.add_task(session_logger.remove)
    return DocumentsSimilarityOutputSchema(
        similar_papers_links=[doc.metadata["source"] for doc in retrieved_docs]
    )
