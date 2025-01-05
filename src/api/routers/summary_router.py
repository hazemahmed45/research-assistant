"""
Summarize Document API Module

This module defines the API endpoint for summarize document.
It provides an endpoint to summarize the document given in the request body

"""

from __future__ import annotations
from typing import List
import sys
from fastapi import APIRouter, BackgroundTasks
import loguru
from loguru import logger
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents.base import Document

from src.misc.document_schema import DocumentStructureSchema
from src.models.doc_embedder import BaseDocumentEmbedder
from src.models.doc_analyizer import BaseDocumentAnalyizer
from src.models.doc_vectorstore import BaseVectorstore
from src.api.config import ApiSettings
from src.api.schema import (
    DocumentSummaryInputSchema,
    DocumentSummaryOutputSchema,
)
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id
from src.models import (
    SingletonModelFactory,
    DocumentAnalyizerSettings,
    DocumentEmbedderSettings,
    VectorstoreSettings,
)
from src.misc.utils import merge_documents
from src.enums import Constants

REQUEST_TYPE = "sum-doc"

api_settings = ApiSettings()
sum_doc_router = APIRouter()

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


@sum_doc_router.post("/sum_doc", tags=[Constants.SUMMARIZE_DOCUMNET_TAG.name])
async def summarize_doc(
    schema: DocumentSummaryInputSchema,
    background_tasks: BackgroundTasks,
) -> DocumentSummaryOutputSchema:
    """
    TODO docstring
    """
    request_unique_id: str = create_unique_user_id()
    session_logger: loguru.Logger = logger.bind(
        user_unique_id=request_unique_id,
        request_type=Constants.SUMMARIZE_DOCUMNET_REQUEST_TYPE.name,
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

    # TODO IMPLEMENT SUMMARIZE DOCUMENTS ROUTE

    pdf_loader = PyPDFLoader(file_path=schema.link)
    document_pages: List[Document] = pdf_loader.load()
    document: Document = merge_documents(documents=document_pages)

    document_structured_extraction: DocumentStructureSchema = document_analyizer(
        doc_context=document
    )
    print(vectorstore.add_structured_document(document_structured_extraction))
    background_tasks.add_task(
        vectorstore.add_structured_document,
        structured_document=document_structured_extraction,
    )
    background_tasks.add_task(session_logger.remove)
    return DocumentSummaryOutputSchema(
        document_summary_extraction=document_structured_extraction
    )
