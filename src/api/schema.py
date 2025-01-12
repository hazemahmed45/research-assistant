"""
API Response Schemas Module
==========================

Defines Pydantic models for API response schemas.

Classes
-------
"""

from typing import List, Optional
from pydantic import BaseModel, Field, AnyUrl
from src.misc.document_schema import DocumentStructureSchema
from src.enums import SimilarityCategories


class DocumentSummaryInputSchema(BaseModel):
    link: AnyUrl = Field(description="link of the paper pdf")


class DocumentSummaryOutputSchema(BaseModel):
    document_summary_extraction: DocumentStructureSchema = Field(
        description="detailed summary extractions of the research paper"
    )


class DocumentsComparisonInputSchema(BaseModel):
    links: List[AnyUrl] = Field(
        default=[], description="links of documents to compare with each other"
    )


class DocumentsComparisonOutputSchema(BaseModel):
    document_summary_extraction: DocumentStructureSchema = Field(
        description="detailed comparison between the summary extractions of all the paper to compare"
    )


class DocumentsSimilarityInputSchema(BaseModel):
    link: AnyUrl = Field(description="link of the paper pdf to find similar papers to")
    topk: int = Field(default=5, description="the number of similar papers to retrieve")
    similarity_category: SimilarityCategories = Field(
        default=SimilarityCategories.SUMMARY,
        description="the choosen category to get similar papers on",
    )
    do_similarity_summary: bool = Field(
        default=False,
        description="whether to return a summary of the similarity between the papers",
    )


class DocumentsSimilarityOutputSchema(BaseModel):
    similar_papers_links: List[AnyUrl] = Field(
        default=[], description="links of the similar papers"
    )
    similarity_summarization: Optional[str] = Field(
        None, description="summarlization of the similarities of the papers"
    )
