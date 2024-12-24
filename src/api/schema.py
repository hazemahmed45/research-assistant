"""
API Response Schemas Module
==========================

Defines Pydantic models for API response schemas.

Classes
-------
"""

from typing import List
from pydantic import BaseModel, Field


class DocumentSummaryInputSchema(BaseModel):
    pass


class DocumentSummaryOutputSchema(BaseModel):
    pass


class DocumentsComparisonInputSchema(BaseModel):
    pass


class DocumentsComparisonOutputSchema(BaseModel):
    pass


class DocumentsSimilarityInputSchema(BaseModel):
    pass


class DocumentsSimilarityOutputSchema(BaseModel):
    pass
