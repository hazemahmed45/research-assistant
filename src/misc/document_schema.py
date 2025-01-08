"""
Utility Functions for Working with Pydantic Schemas
===================================================

This module provides functions to generate schema representations, create complaint schema modules from JSON structures, and defines a multiple complaint schema class for handling lists of typed complaints.

"""

from __future__ import annotations
from typing import List, Optional, Union
from typing_extensions import Annotated
import datetime
from datetime import date
from pydantic import BaseModel, Field, AnyUrl
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import PlainValidator
from src.enums import Constants


def url_serializer_fn(x: AnyUrl) -> str:
    return str(x)


def date_serializer_fn(x: date) -> str:
    return x.strftime(Constants.DATE_FORMAT.value)


def date_validator_fn(x: Union[str, date]) -> date | None:
    if isinstance(x, str):
        return datetime.datetime.strptime(x, Constants.DATE_FORMAT.value)
    elif isinstance(x, date):
        return x
    else:
        return None


class DocumentStructureSchema(BaseModel):
    """Document attributes"""

    id: str = Field(
        default="",
        description="the unique id of the paper",
    )
    link: (
        Annotated[AnyUrl, PlainSerializer(url_serializer_fn, return_type=str)] | None
    ) = Field(default=None, description="the url of the paper")
    title: str = Field(
        default="",
        description="the headline or name of the paper that concisely conveys the essence of the study, usually the heading of the study in the first lines",
    )
    publication_date: (
        Annotated[
            date,
            PlainSerializer(date_serializer_fn, return_type=str),
            PlainValidator(date_validator_fn),
        ]
        | None
    ) | None = Field(
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
        description="the individuals who have contributed significantly to the study. this should be listed as a list seperated by commas and finished by and endline. usually authors of a study is after the title",
    )
    domain: List[str] | None = Field(
        default=[],
        description="the domain of the paper, which identify the specific area or field of study to which the research belongs. It represents the subject matter, discipline, or context in which the research is conducted and where it contributes new knowledge.usally the domain and tags of the study can be extracted or directly specified in the abstraction, also this should be listed as a list seperated by commas and finished by and endline",
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
    o = DocumentStructureSchema(repo="HERE")
    # print(o)
    # o.model_fields["repo"] = "here"
    # print(
    #     DocumentStructureSchema.model_fields["authors_name"].annotation
    #     == Optional[List[str]]
    # )
    print(o.model_dump()["repo"])
