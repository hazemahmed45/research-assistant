"""
Document Analyizer Module
===========================

This module encompasses various classes for document analyizer , 
utilizing different serving types and models.
"""

from __future__ import annotations
import os
import warnings
from abc import abstractmethod
from typing import Any, Optional, Union, List
from enum import IntEnum

# import json
import time

# from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from transformers import pipeline
from langchain_core.prompts import (
    PromptTemplate,
    # FewShotPromptTemplate,
    # ChatPromptTemplate,
    # FewShotChatMessagePromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser

# from langchain_core.messages import SystemMessage
# from langchain_core.prompt_values import StringPromptValue
# from langchain_core.runnables import RunnableLambda
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.language_models.base import BaseLanguageModel

# from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.llms.vllm import VLLMOpenAI

# from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface.llms import HuggingFaceEndpoint, HuggingFacePipeline

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from gradio_client import Client

from src.enums import LLMModelCards, LLMServingType
from src.misc.document_schema import DocumentStructureSchema

from src.misc.utils import remove_punctuations


class DocumentAnalyizerSettings(BaseSettings):
    """
    **Configuration settings for the Complain Recognition Model**

    :param llm_api_url: URL where the model is hosted.
    :type llm_api_url: str
    :param model_card: LLM model card from Hugging Face Hub.
    :type model_card: LLMModelCards
    :param max_new_tokens: Maximum new tokens for model generation.
    :type max_new_tokens: int
    :param top_k: Top K for model generation (optional).
    :type top_k: Union[float, None]
    :param top_p: Top P for model generation (optional).
    :type top_p: Union[float, None]
    :param temperature: Temperature for model generation.
    :type temperature: float
    :param num_return_sequences: Number of return sequences for model generation.
    :type num_return_sequences: int
    :param do_sample: Whether to sample during model generation.
    :type do_sample: bool
    :param max_time: Maximum time for model generation (optional).
    :type max_time: Union[float, None]
    :param serving_type: Type of serving for the LLM.
    :type serving_type: LLMServingType
    :param prompt_preprocessing: Whether to preprocess prompt.
    :type prompt_preprocessing: bool
    """

    llm_api_url: str = Field(
        default="https://api-inference.huggingface.co/models/",
        description="the url the model is hosted on",
    )
    model_card: LLMModelCards = Field(
        default=LLMModelCards.MISTRAL_7B_INSTRUCT_V02,
        description="the llm model card from huggingface hub",
    )
    max_new_tokens: int = Field(default=200)
    top_k: Union[float, None] = Field(default=None)
    top_p: Union[float, None] = Field(default=None)
    temperature: float = Field(default=0.8)
    num_return_sequences: int = Field(default=1)
    do_sample: bool = Field(default=True)
    max_time: Union[float, None] = Field(default=None)
    serving_type: LLMServingType = Field(default=LLMServingType.HUGGINGFACE)
    prompt_preprocessing: bool = Field(default=True)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


class DocumentAnalyizer:
    """
    **Abstract base class for document analyizer**
    """

    def __init__(self) -> None:
        """**Initialize the document analyizer instance**"""
        pass

    @abstractmethod
    def __call__(self, document_text: str) -> DocumentStructureSchema:
        """
        **Process a document text**

        :param document_text: The documenttext to process.
        :type document_text: str
        :return: Processed result.
        :rtype: DocumentStructureSchema
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError(f"__call__ is not implemented for {self.__class__}")


class DocumentAnalyizerLLM(DocumentAnalyizer):
    """
    **document analyizer using a Large Language Model (LLM)**

    :param prompt_preprocessing: Whether to preprocess prompt.
    :type prompt_preprocessing: bool
    """

    class PromptInputs(IntEnum):
        CONTEXT = 0
        QUESTION = 1
        MOTIVATION = 2
        PROBLEMS = 3
        CHALLENGES = 4
        CONTRIBUTION = 5
        TECHNIQUES = 6
        DATASETS = 7
        METHODOLOGY = 8
        PROPOSED_MODEL = 9
        RESULTS = 10
        FUTURE_WORK = 11

    def __init__(
        self,
        prompt_preprocessing: bool = True,
    ) -> None:
        """
        **Initialize the LLM-based document analyizer**

        :param prompt_preprocessing: Whether to preprocess prompt.
        :type prompt_preprocessing: bool
        """
        super().__init__()
        self._arg_parameters = {}
        self._prompt_preprocessing = prompt_preprocessing

    def __call__(self, doc_context: str) -> DocumentStructureSchema:
        """
        **Process a doc context using the LLM**

        :param doc_context: The doc context text.
        :type doc_context: str
        :return: Processed result.
        :rtype: DocumentStructureSchema
        """
        analized_document = DocumentStructureSchema()
        if self._prompt_preprocessing:
            doc_context = remove_punctuations(doc_context, exclude=",.").lower()
        doc_attr_analyizer_chain = (
            self.get_prompt(is_attribute=True) | self.get_llm() | StrOutputParser()
        )
        doc_summarizer_chain = (
            self.get_prompt(is_attribute=False) | self.get_llm() | StrOutputParser()
        )
        for attr_name, attr_metadata in DocumentStructureSchema.model_json_schema()[
            "properties"
        ].items():
            if not any([attr_name == key for key in ["id", "link", "summary"]]):
                attr_desc = attr_metadata["description"]
                question = "provide me with " + attr_desc

                analysis_res: str = doc_attr_analyizer_chain.invoke(
                    {
                        self.get_inputs()[self.PromptInputs.CONTEXT.value]: doc_context,
                        self.get_inputs()[self.PromptInputs.QUESTION.value]: question,
                    }
                )
                if (
                    analized_document.model_fields[attr_name].annotation
                    == Optional[List[str]]
                ):
                    analysis_res = analysis_res.split(",")
                analized_document.model_fields[attr_name] = analysis_res
        analized_document.summary = doc_summarizer_chain.invoke(
            {
                self.get_inputs()[
                    self.PromptInputs.MOTIVATION.value
                ]: analized_document.motivation,
                self.get_inputs()[
                    self.PromptInputs.PROBLEMS.value
                ]: analized_document.problems,
                self.get_inputs()[
                    self.PromptInputs.CHALLENGES.value
                ]: analized_document.challenges,
                self.get_inputs()[
                    self.PromptInputs.CONTRIBUTION.value
                ]: analized_document.contribution,
                self.get_inputs()[
                    self.PromptInputs.TECHNIQUES.value
                ]: analized_document.techniques,
                self.get_inputs()[self.PromptInputs.DATASETS.value]: ", ".join(
                    analized_document.datasets
                ),
                self.get_inputs()[
                    self.PromptInputs.METHODOLOGY.value
                ]: analized_document.methodology,
                self.get_inputs()[
                    self.PromptInputs.PROPOSED_MODEL.value
                ]: analized_document.proposed_model,
                self.get_inputs()[
                    self.PromptInputs.RESULTS.value
                ]: analized_document.results,
                self.get_inputs()[
                    self.PromptInputs.FUTURE_WORK.value
                ]: analized_document.future_work,
            }
        )
        return analized_document

    def get_llm(self) -> BaseLanguageModel:
        """
        **Get the Large Language Model instance**

        :return: The LLM instance.
        :rtype: BaseLanguageModel
        :raises NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError(
            "This function is to be implemented in a more concrete class"
        )

    def get_prompt(self, is_attribute: bool = True) -> PromptTemplate:
        """
        **Get the prompt template for the LLM**

        :return: The prompt template.
        :rtype: PromptTemplate
        """

        return PromptTemplate.from_template(
            template=f"{self.get_prompt_prefix(is_attribute=is_attribute)}\
                {self.get_prompt_suffix(is_attribute=is_attribute)}",
            template_format="jinja2",
        )

    def get_prompt_prefix(self, is_attribute: bool = True) -> str:
        if is_attribute:
            return "you are an expert data analyizer and you are instructed to extract the answer for following question given the following context\n\n"
        else:
            return "you are an expert data analyizer and you are instructed to summarize all the following contexts of a research paper in one paragraph summarizing all the important information\n\n"

    def get_prompt_suffix(self, is_attribute: bool = True) -> str:
        if is_attribute:
            return "##########################\ncontext: {{context}}\n\nquestion: {{question}}\n##########################\nextracted answer: "
        else:
            return "##########################\nmotivation: {{motivation}}\n\n\
            problems: {{problems}}\n\nchallenges: {{challenges}}\n\n\
                contribution: {{contribution}}\n\ntechniques: {{techniques}}\n\n\
                    datasets: {{datasets}}\n\nmethodology: {{methodology}}\n\n\
                        proposed model: {{proposed_model}}\n\nresults: {{results}}\n\n\
                            future work: {{future_work}}\n##########################\nsummary: "

    def get_inputs(self) -> List[str]:
        return [
            "context",
            "question",
            "motivation",
            "problems",
            "challenges",
            "contribution",
            "technqiues",
            "datasets",
            "methodology",
            "proposed_model",
            "results",
            "future_work",
        ]

    def set_args(self, **kwargs):
        """
        **Set additional arguments for the document analyizer**

        :param kwargs: Additional keyword arguments.
        """
        self._arg_parameters = kwargs


class DocumentAnalyizerPipelineLLM(DocumentAnalyizerLLM):
    """
    **document analyizer using a pipeline LLM, runs locally**

    :param model_name: Name of the LLM model.
    :type model_name: str
    :param prompt_preprocessing: Whether to preprocess document text.
    :type prompt_preprocessing: bool
    :param return_full_text: Whether to return the full text.
    :type return_full_text: bool
    :param use_cache: Whether to use caching.
    :type use_cache: bool
    :param max_new_tokens: Maximum new tokens for generation.
    :type max_new_tokens: int
    :param top_k: Top K for generation (optional).
    :type top_k: int
    :param top_p: Top P for generation (optional).
    :type top_p: int
    :param temperature: Temperature for generation.
    :type temperature: float
    :param repetition_penalty: Repetition penalty for generation (optional).
    :type repetition_penalty: float
    :param num_return_sequences: Number of return sequences.
    :type num_return_sequences: int
    :param do_sample: Whether to sample during generation.
    :type do_sample: bool
    :param max_time: Maximum time for generation (optional).
    :type max_time: float
    :param device: Device to use (e.g., "cuda").
    :type device: str
    """

    def __init__(
        self,
        model_name: str,
        auth_token: str,
        prompt_preprocessing: bool = True,
        return_full_text: bool = False,
        use_cache=True,
        max_new_tokens: int = 100,
        top_k: int = None,
        top_p: int = None,
        temperature: float = 0.5,
        repetition_penalty: float = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        max_time: float = None,
        device: str = "cuda",
    ) -> None:
        """
        **Initialize the pipeline LLM-based document analyizer**


        :param model_name: Name of the LLM model.
        :type model_name: str
        :param prompt_preprocessing: Whether to preprocess document text.
        :type prompt_preprocessing: bool
        :param return_full_text: Whether to return the full text.
        :type return_full_text: bool
        :param use_cache: Whether to use caching.
        :type use_cache: bool
        :param max_new_tokens: Maximum new tokens for generation.
        :type max_new_tokens: int
        :param top_k: Top K for generation (optional).
        :type top_k: int
        :param top_p: Top P for generation (optional).
        :type top_p: int
        :param temperature: Temperature for generation.
        :type temperature: float
        :param repetition_penalty: Repetition penalty for generation (optional).
        :type repetition_penalty: float
        :param num_return_sequences: Number of return sequences.
        :type num_return_sequences: int
        :param do_sample: Whether to sample during generation.
        :type do_sample: bool
        :param max_time: Maximum time for generation (optional).
        :type max_time: float
        :param device: Device to use (e.g., "cuda").
        :type device: str
        """
        super().__init__(
            prompt_preprocessing=prompt_preprocessing,
        )
        self.set_args(
            **{
                "return_full_text": return_full_text,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                "max_time": max_time,
            }
        )
        pipe = pipeline(
            task="text-generation", model=model_name, device=device, token=auth_token
        )
        self.arg_parameters = {**self.arg_parameters, "use_cache": use_cache}
        self.llm = HuggingFacePipeline(
            pipeline=pipe, pipeline_kwargs=self.arg_parameters
        )

    def get_llm(self) -> BaseLanguageModel:
        """
        **Get the pipeline LLM instance**

        :return: The pipeline LLM instance.
        :rtype: BaseLanguageModel
        """
        return self.llm


class DocumentAnalyizerAPILLM(DocumentAnalyizerLLM):
    """
    **Handles document analyizer using an API-connected Large Language Model (LLM), communicate with deployed LLM on cloud**

    :param llm_api_url: URL of the LLM API.
    :type llm_api_url: str
    :param auth_token: Authentication token for the LLM API.
    :type auth_token: str
    :param prompt_preprocessing: Whether to preprocess prompt, defaults to True.
    :type prompt_preprocessing: bool, optional
    """

    def __init__(
        self,
        llm_api_url: str,
        auth_token: str,
        prompt_preprocessing: bool = True,
    ) -> None:
        """
        **Initializes the API LLM for document analyizer**

        :param llm_api_url: URL of the LLM API.
        :type llm_api_url: str
        :param auth_token: Authentication token for the LLM API.
        :type auth_token: str
        :param prompt_preprocessing: Whether to preprocess prompt, defaults to True.
        :type prompt_preprocessing: bool, optional
        """
        super().__init__(
            prompt_preprocessing=prompt_preprocessing,
        )
        self.llm_api_url = llm_api_url
        self.headers = {"Authorization": auth_token}


class DocumentAnalyizerHuggingFaceAPILLM(DocumentAnalyizerAPILLM):
    """
    **Utilizes Hugging Face's Inference Client API for document analyizer with LLMs**

    :param llm_api_url: URL of the LLM API.
    :type llm_api_url: str
    :param auth_token: Authentication token for the LLM API.
    :type auth_token: str
    :param model_name: name of the model in the hub
    :type model_name: str
    :param prompt_preprocessing: Whether to preprocess prompt, defaults to True.
    :type prompt_preprocessing: bool, optional
    :param return_full_text: Whether to return the full text response, defaults to False.
    :type return_full_text: bool, optional
    :param use_cache: Whether to use caching, defaults to False.
    :type use_cache: bool, optional
    :param max_new_tokens: Maximum new tokens to generate, defaults to 200.
    :type max_new_tokens: int, optional
    :param top_k: Top k results to consider, defaults to None.
    :type top_k: int, optional
    :param top_p: Top p results to consider, defaults to None.
    :type top_p: int, optional
    :param temperature: Temperature for generation, defaults to 0.5.
    :type temperature: float, optional
    :param repetition_penalty: Penalty for repetition, defaults to None.
    :type repetition_penalty: float, optional
    :param num_return_sequences: Number of sequences to return, defaults to 1.
    :type num_return_sequences: int, optional
    :param do_sample: Whether to sample, defaults to True.
    :type do_sample: bool, optional
    :param max_time: Maximum time in seconds, defaults to 10000.
    :type max_time: float, optional
    """

    def __init__(
        self,
        llm_api_url: str,
        auth_token: str,
        model_name: str,
        prompt_preprocessing: bool = True,
        return_full_text: bool = False,
        use_cache=False,
        max_new_tokens: int = 200,
        top_k: int = None,
        top_p: int = None,
        temperature: float = 0.5,
        repetition_penalty: float = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        max_time: float = 10000,
        **kwargs,
    ) -> None:
        """
        **Initializes the Hugging Face's API LLM for document analyizer**

        :param llm_api_url: URL of the LLM API.
        :type llm_api_url: str
        :param auth_token: Authentication token for the LLM API.
        :type auth_token: str
        :param model_name: name of the model in the hub
        :type model_name: str
        :param prompt_preprocessing: Whether to preprocess prompt, defaults to True.
        :type prompt_preprocessing: bool, optional
        :param return_full_text: Whether to return the full text response, defaults to False.
        :type return_full_text: bool, optional
        :param use_cache: Whether to use caching, defaults to False.
        :type use_cache: bool, optional
        :param max_new_tokens: Maximum new tokens to generate, defaults to 200.
        :type max_new_tokens: int, optional
        :param top_k: Top k results to consider, defaults to None.
        :type top_k: int, optional
        :param top_p: Top p results to consider, defaults to None.
        :type top_p: int, optional
        :param temperature: Temperature for generation, defaults to 0.5.
        :type temperature: float, optional
        :param repetition_penalty: Penalty for repetition, defaults to None.
        :type repetition_penalty: float, optional
        :param num_return_sequences: Number of sequences to return, defaults to 1.
        :type num_return_sequences: int, optional
        :param do_sample: Whether to sample, defaults to True.
        :type do_sample: bool, optional
        :param max_time: Maximum time in seconds, defaults to 10000.
        :type max_time: float, optional
        """
        super().__init__(
            llm_api_url=llm_api_url,
            auth_token=auth_token,
            prompt_preprocessing=prompt_preprocessing,
        )
        self.set_args(
            **{
                "return_full_text": return_full_text,
                "max_new_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "model_kwargs": {
                    "max_time": max_time,
                    "num_return_sequences": num_return_sequences,
                },
            }
        )

        self.llm = HuggingFaceEndpoint(
            # repo_id=model_name,
            endpoint_url=f"{llm_api_url}{model_name}",
            huggingfacehub_api_token=auth_token,
            cache=use_cache,
            **self._arg_parameters,
        )

    def get_llm(self) -> BaseLanguageModel:
        """
        **Retrieves the configured Huggingface LLM**

        :return: The BaseLanguageModel instance.
        :rtype: BaseLanguageModel
        """
        return self.llm

    def __call__(self, doc_context: str) -> DocumentStructureSchema:
        """
        **Process a document text using the LLM**

        :param doc_context: The document context text.
        :type doc_context: str
        :return: Processed result.
        :rtype: DocumentStructureSchema
        """
        result = None
        try:
            result: DocumentStructureSchema = super().__call__(doc_context)
        except HfHubHTTPError as e:
            if e is not None and "Rate" in e.server_message:
                warnings.warn(f"{e}, waiting for 10 mins and recall")
                time.sleep(60 * 10)
                result = self.__call__(doc_context)
            else:
                raise e
        return result


class DocumentAnalyizerVLLMOpenAIAPILLM(DocumentAnalyizerAPILLM):
    """
    **Utilizes LLMs hosted using VLLM inference server for document analyizer**


    :param llm_api_url: URL of the LLM API.
    :type llm_api_url: str
    :param auth_token: Authentication token for the LLM API.
    :type auth_token: str
    :param model_name: name of the model in the hub
    :type model_name: str
    :param prompt_preprocessing: Whether to preprocess prompt, defaults to True.
    :type prompt_preprocessing: bool, optional
    :param return_full_text: Whether to return the full text response, defaults to False.
    :type return_full_text: bool, optional
    :param use_cache: Whether to use caching, defaults to False.
    :type use_cache: bool, optional
    :param max_new_tokens: Maximum new tokens to generate, defaults to 200.
    :type max_new_tokens: int, optional
    :param top_k: Top k results to consider, defaults to None.
    :type top_k: int, optional
    :param top_p: Top p results to consider, defaults to None.
    :type top_p: int, optional
    :param temperature: Temperature for generation, defaults to 0.5.
    :type temperature: float, optional
    :param repetition_penalty: Penalty for repetition, defaults to None.
    :type repetition_penalty: float, optional
    :param num_return_sequences: Number of sequences to return, defaults to 1.
    :type num_return_sequences: int, optional
    :param do_sample: Whether to sample, defaults to True.
    :type do_sample: bool, optional
    :param max_time: Maximum time in seconds, defaults to 10000.
    :type max_time: float, optional
    :param presence_penalty: penalty on the presence of same token multiple times, defaults to 1
    :type presence_penalty: float, optional
    """

    def __init__(
        self,
        llm_api_url: str,
        auth_token: str,
        model_name: str,
        prompt_preprocessing: bool = True,
        use_cache=False,
        max_new_tokens: int = 200,
        top_p: int = None,
        temperature: float = 0.5,
        repetition_penalty: float = None,
        num_return_sequences: int = 1,
        max_time: float = None,
        presence_penalty: float = 1,
        **kwargs,
    ) -> None:
        """
        **Initializes VLLM instance for interactive document analyizer**

        :param llm_api_url: URL of the LLM API.
        :type llm_api_url: str
        :param auth_token: Authentication token for the LLM API.
        :type auth_token: str
        :param model_name: name of the model in the hub
        :type model_name: str
        :param prompt_preprocessing: Whether to preprocess prompt, defaults to True.
        :type prompt_preprocessing: bool, optional
        :param return_full_text: Whether to return the full text response, defaults to False.
        :type return_full_text: bool, optional
        :param use_cache: Whether to use caching, defaults to False.
        :type use_cache: bool, optional
        :param max_new_tokens: Maximum new tokens to generate, defaults to 200.
        :type max_new_tokens: int, optional
        :param top_k: Top k results to consider, defaults to None.
        :type top_k: int, optional
        :param top_p: Top p results to consider, defaults to None.
        :type top_p: int, optional
        :param temperature: Temperature for generation, defaults to 0.5.
        :type temperature: float, optional
        :param repetition_penalty: Penalty for repetition, defaults to None.
        :type repetition_penalty: float, optional
        :param num_return_sequences: Number of sequences to return, defaults to 1.
        :type num_return_sequences: int, optional
        :param do_sample: Whether to sample, defaults to True.
        :type do_sample: bool, optional
        :param max_time: Maximum time in seconds, defaults to 10000.
        :type max_time: float, optional
        :param presence_penalty: penalty on the presence of same token multiple times, defaults to 1
        :type presence_penalty: float, optional
        """
        super().__init__(
            llm_api_url=llm_api_url,
            auth_token=auth_token,
            prompt_preprocessing=prompt_preprocessing,
        )
        self.set_args(
            **{
                "max_tokens": max_new_tokens,
                "top_p": top_p if top_p is not None else 1.0,
                "temperature": temperature,
                "frequency_penalty": (
                    repetition_penalty if repetition_penalty is not None else 1.0
                ),
                "n": num_return_sequences,
                "timeout": max_time,
                "presence_penalty": presence_penalty,
            }
        )
        self.llm = VLLMOpenAI(
            base_url=llm_api_url,
            api_key=auth_token,
            model_name=model_name,
            cache=use_cache,
            **self._arg_parameters,
        )

    def get_llm(self) -> BaseLanguageModel:
        """
        **Retrieves the configured VLLM LLM**

        :return: The BaseLanguageModel instance.
        :rtype: BaseLanguageModel
        """
        return self.llm


if __name__ == "__main__":

    settings = DocumentAnalyizerSettings()
    cr = DocumentAnalyizerHuggingFaceAPILLM(
        llm_api_url=settings.llm_api_url,
        auth_token="",
        model_name=LLMModelCards.MISTRAL_7B_INSTRUCT_V03.value,
        max_new_tokens=settings.max_new_tokens,
        top_k=settings.top_k,
        top_p=settings.top_p,
        do_sample=settings.do_sample,
        temperature=settings.temperature,
        num_return_sequences=settings.num_return_sequences,
        max_time=settings.max_time,
        prompt_preprocessing=settings.prompt_preprocessing,
    )
    # cr.get_prompt()
    print(cr.get_prompt().format())
