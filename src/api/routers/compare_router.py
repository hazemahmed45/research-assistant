"""
Compare Document API Module

This module defines the API endpoint for compare document.
It provides an endpoint to compare documents in the request body with each others

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
    DocumentsComparisonInputSchema,
    DocumentsComparisonOutputSchema,
)
from src.misc.logger_handlers import FileHandler
from src.misc.create_unique_id import create_unique_user_id
from src.misc.utils import merge_pages_in_document
from src.enums import Constants


api_settings = ApiSettings()
comp_doc_router = APIRouter()
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


@comp_doc_router.post("/comp_doc", tags=["Compare Document"])
async def compare_doc(
    schema: DocumentsComparisonInputSchema,
    background_tasks: BackgroundTasks,
) -> DocumentsComparisonOutputSchema:
    """
    TODO docstring
    """
    request_unique_id: str = create_unique_user_id()
    session_logger: loguru.Logger = logger.bind(
        user_unique_id=request_unique_id,
        request_type=Constants.COMPARE_DOCUMENTS_REQUEST_TYPE.value,
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

    # TODO IMPLEMENT COMPARE DOCUMENTS ROUTE
    structured_docs: List[DocumentStructureSchema] = []
    for ii, paper_link in enumerate(schema.links):
        structured_doc: Union[DocumentStructureSchema, None] = (
            database.get_structured_document_by_link(document_link=str(paper_link))
        )
        if structured_doc == None:
            pdf_loader = PyPDFLoader(file_path=paper_link)
            document_pages: List[Document] = pdf_loader.load()
            session_logger.info(f"Loaded {len(document_pages)} pages")
            document: Document = merge_pages_in_document(document_pages=document_pages)

            session_logger.info(
                f"Merged all pages into one document with id {document.id}"
            )

            session_logger.info("Start document analyizing")
            structured_doc: DocumentStructureSchema = document_analyizer(
                doc_context=document, task=document_analyizer.AnalyizingTask.SUMMARY
            )
            session_logger.info("Finished document analyizing")
            background_tasks.add_task(
                vectorstore.add_structured_document,
                structured_document=structured_doc,
            )
            background_tasks.add_task(
                database.push_document, structured_document=structured_doc
            )
        structured_docs.append(structured_doc)
    comparison_summarization = document_analyizer(
        doc_context="\n\n".join(
            [
                f"Context {ii}:\n" + doc.model_dump()[schema.comparison_category.value]
                for ii, doc in enumerate(structured_docs, start=1)
            ]
        ),
        task=document_analyizer.AnalyizingTask.COMPARE,
    ).summary
    background_tasks.add_task(session_logger.remove)
    return DocumentsComparisonOutputSchema(
        comparison_summarization=comparison_summarization
    )
