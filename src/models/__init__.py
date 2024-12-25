"""Research Assistant Model Factory

This module provides a factory for creating various research assistant models.
"""

from typing import Union
from pydantic_settings import BaseSettings
from src.enums import (
    LLMServingType,
)
from src.models.doc_analyizer import (
    DocumentAnalyizerHuggingFaceAPILLM,
    DocumentAnalyizerPipelineLLM,
    DocumentAnalyizerVLLMOpenAIAPILLM,
    DocumentAnalyizerSettings,
)

from src.misc.exceptions import (
    ModelServingTypeNotSupported,
)


class ModelFactory:
    """Research Assistant Model Factory

    This class provides a factory for creating data analyizer.
    """

    @staticmethod
    def build_model(
        settings: BaseSettings, authentication_token: str, cache_dir: str = None
    ) -> DocumentAnalyizerHuggingFaceAPILLM | DocumentAnalyizerVLLMOpenAIAPILLM | None:
        """Model factory for data analyizer

        :param settings: Model configuration settings
        :type settings: BaseSettings
        :param authentication_token: Authentication token for model APIs
        :type authentication_token: str
        :param cache_dir: Directory for caching model downloads (optional)
        :type cache_dir: str
        :return: The built data analyizer model based on the settings passed
        :rtype:  Union[ DocumentAnalyizerHuggingFaceAPILLM, DocumentAnalyizerVLLMOpenAIAPILLM ,DocumentAnalyizerVLLMOpenAIAPILLM]
        :raises ModelInferenceTypeNotSupported: If the model inference type is not supported
        """

        if isinstance(settings, DocumentAnalyizerSettings):
            return ModelFactory._get_document_analyizer(
                settings=settings,
                authentication_token=authentication_token,
            )

    @staticmethod
    def _get_document_analyizer(
        settings: DocumentAnalyizerSettings, authentication_token: str
    ) -> Union[
        DocumentAnalyizerHuggingFaceAPILLM,
        DocumentAnalyizerVLLMOpenAIAPILLM,
        DocumentAnalyizerPipelineLLM,
    ]:
        """Retrieve a data analyizer API Model

        :param settings: data analyizer model configuration settings
        :type settings: DocumentAnalyizerSettings
        :param authentication_token: Authentication token for model APIs
        :type authentication_token: str
        :return: The data analyizer API model
        :rtype: Union[ DocumentAnalyizerHuggingFaceAPILLM, DocumentAnalyizerVLLMOpenAIAPILLM ,DocumentAnalyizerVLLMOpenAIAPILLM]
        :raises ModelServingTypeNotSupported: If the serving type is not supported
        """
        if not any([s_type == settings.serving_type for s_type in LLMServingType]):
            raise ValueError(f"{settings.serving_type} is not supported in this repo")
        if settings.serving_type == LLMServingType.VLLM:
            return DocumentAnalyizerVLLMOpenAIAPILLM(
                llm_api_url=settings.llm_api_url,
                auth_token=authentication_token,
                model_name=settings.model_card.value,
                max_new_tokens=settings.max_new_tokens,
                top_p=settings.top_p,
                temperature=settings.temperature,
                num_return_sequences=settings.num_return_sequences,
                max_time=settings.max_time,
                prompt_preprocessing=settings.prompt_preprocessing,
            )
        elif settings.serving_type == LLMServingType.HUGGINGFACE_API:
            return DocumentAnalyizerHuggingFaceAPILLM(
                llm_api_url=settings.llm_api_url,
                auth_token=authentication_token,
                model_name=settings.model_card.value,
                max_new_tokens=settings.max_new_tokens,
                top_k=settings.top_k,
                top_p=settings.top_p,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
                num_return_sequences=settings.num_return_sequences,
                max_time=settings.max_time,
                prompt_preprocessing=settings.prompt_preprocessing,
            )
        elif settings.serving_type == LLMServingType.HUGGINGFACE:
            return DocumentAnalyizerPipelineLLM(
                model_name=settings.model_card.value,
                prompt_preprocessing=settings.prompt_preprocessing,
                auth_token=authentication_token,
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
                "Selected model serving is not supported"
            )


if __name__ == "__main__":
    pass
