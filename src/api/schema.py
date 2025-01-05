"""
API Response Schemas Module
==========================

Defines Pydantic models for API response schemas.

Classes
-------
"""

from typing import List
from pydantic import BaseModel, Field, AnyUrl
from src.misc.document_schema import DocumentStructureSchema


class DocumentSummaryInputSchema(BaseModel):
    link: AnyUrl = Field(description="link of the pdf of the paper")


class DocumentSummaryOutputSchema(BaseModel):
    document_summary_extraction: DocumentStructureSchema = Field(
        description="detailed summary extractions of the research paper"
    )


class DocumentsComparisonInputSchema(BaseModel):
    pass


class DocumentsComparisonOutputSchema(BaseModel):
    pass


class DocumentsSimilarityInputSchema(BaseModel):
    pass


class DocumentsSimilarityOutputSchema(BaseModel):
    pass
