"""
Utility Functions for Working with Pydantic Schemas
===================================================

This module provides functions to generate schema representations, create complaint schema modules from JSON structures, and defines a multiple complaint schema class for handling lists of typed complaints.

"""

from __future__ import annotations
from typing import List, Optional
from datetime import date
from uuid import uuid4
from pydantic import UUID4, BaseModel, Field, AnyUrl


class DocumentStructureSchema(BaseModel):
    """Document attributes"""

    id: UUID4 = Field(
        default_factory=uuid4,
        description="the unique id of the paper",
    )
    link: AnyUrl | None = Field(
        default=None,
        description="the url of the paper",
    )
    title: str = Field(
        default="",
        description="the headline or name of the paper that concisely conveys the essence of the study",
    )
    publication_date: date | None = Field(
        default=None,
        description="the date were the paper is officially published, this should be formated as follows DD-MM-YYYY",
    )
    summary: str | None = Field(
        default="",
        description="the summary of the paper taking in consideration the context of the motivation, problems, challenges, contribution, technqiues, datasets, methodology, proposed model, results, and future work",
    )
    contribution: str | None = Field(
        default="",
        description="the contribution of the paper, which describes the unique value that the research brings to the field.",
    )
    authors_name: List[str] | None = Field(
        default="",
        description="the individuals who have contributed significantly to the study. this should be listed as a list seperated by commas and finished by and endline.",
    )
    domain: List[str] | None = Field(
        default=[],
        description="the domain of the paper, which identify the specific area or field of study to which the research belongs. It represents the subject matter, discipline, or context in which the research is conducted and where it contributes new knowledge. this should be listed as a list seperated by commas and finished by and endline",
    )
    motivation: str | None = Field(
        default="",
        description="the motivation of the paper, which explains the driving force behind the study or why the research is being conducted and why the problem being addressed is important.",
    )
    problems: str | None = Field(
        default="",
        description="the problems the paper is trying to solve, which focuses on identifying and defining the specific issue or gap that the research aims to address.",
    )
    challenges: str | None = Field(
        default="",
        description="the challenges the paper faced, which addresses the difficulties, obstacles, or limitations encountered during the study.",
    )
    techniques: str | None = Field(
        default="",
        description="the techniques the paper used which refers to the methods, tools, algorithms, or procedures that the researchers used to conduct their study or solve the problem they were investigating.",
    )
    datasets: List[str] | None = Field(
        default=[],
        description="the list about the datasets used in the paper, this should be listed as a list seperated by commas and finished by and endline",
    )
    methodology: str | None = Field(
        default="",
        description="the methodolgy the paper used, explaining how the research was conducted. This should provides a detailed description of the methods, tools, and procedures the researcher used to gather, analyze, and interpret data.",
    )
    proposed_model: str | None = Field(
        default="",
        description="the proposed model in the paper which describes a new system, framework, theory, algorithm, or method introduced by the researchers to address a specific problem or question.",
    )
    results: str | None = Field(
        default="",
        description="the findings of the study as presented in a clear and objective manner. It should outlines the outcomes of the research without interpretation or discussion.",
    )
    future_work: str | None = Field(
        default="",
        description="the future potential directions for further research, based on the findings, limitations, or unresolved questions from the current study.",
    )
    venue: str | None = Field(
        default="",
        description="the platform, event, or publication where the research paper is presented, published, or disseminated. should be none if none were fould",
    )
    repo: str | None = Field(
        default="",
        description="the url of the repository the paper code was pushed to. should be none if none were fould",
    )


if __name__ == "__main__":
    o = DocumentStructureSchema()
    print(o)
    o.model_fields["repo"] = "here"
    print(o.model_fields["authors_name"].annotation == Optional[List[str]])
    # print(o.model_fields)