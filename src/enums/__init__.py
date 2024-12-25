"""
Model and Enum Definitions Module
================================

Defines various enums for model cards, dataset cards, inference types, and other settings.

Enums
-----
"""

from enum import Enum


class LLMModelCards(Enum):
    """
    Enum for LLM model cards

    **Values**

    * **ZEPHYR_7B_BETA**: ``HuggingFaceH4/zephyr-7b-beta``
    * **MIXTRAL_8x7B_INSTRUCT_V01**: ``mistralai/Mixtral-8x7B-Instruct-v0.1``
    * **MISTRAL_7B_INSTRUCT_V01**: ``mistralai/Mistral-7B-Instruct-v0.1``
    * **MISTRAL_7B_INSTRUCT_V02**: ``mistralai/Mistral-7B-Instruct-v0.2``
    * **MISTRAL_7B_INSTRUCT_V03**: ``mistralai/Mistral-7B-Instruct-v0.3``
    * **MISTRAL_NEMO_INSTRUCT_2407**: ``mistralai/Mistral-Nemo-Instruct-2407``
    * **MISTRAL_NEMO_MINITRON_8B_BASE**: ``nvidia/Mistral-NeMo-Minitron-8B-Base``
    * **META_LLAMA_3_8B_INSTRUCT**: ``meta-llama/Meta-Llama-3-8B-Instruct``
    * **META_LLAMA_3_1_8B_INSTRUCT**: ``meta-llama/Meta-Llama-3.1-8B-Instruct``
    * **GEMMA_7B**: ``google/gemma-7b``
    * **GEMMA_11_7B_IT**: ``google/gemma-1.1-7b-it``
    * **GEMMA_2B**: ``google/gemma-2b``
    * **GEMMA_11_2B_IT**: ``google/gemma-1.1-2b-it``
    * **PHI_3_MINI_4K_INSTRUCT**: ``microsoft/Phi-3-mini-4k-instruct``
    * **FALCON_MAMBA_7B**: ``tiiuae/falcon-mamba-7b``
    """

    ZEPHYR_7B_BETA = "HuggingFaceH4/zephyr-7b-beta"
    MIXTRAL_8x7B_INSTRUCT_V01 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MISTRAL_7B_INSTRUCT_V01 = "mistralai/Mistral-7B-Instruct-v0.1"
    MISTRAL_7B_INSTRUCT_V02 = "mistralai/Mistral-7B-Instruct-v0.2"
    MISTRAL_7B_INSTRUCT_V03 = "mistralai/Mistral-7B-Instruct-v0.3"
    MISTRAL_NEMO_INSTRUCT_2407 = "mistralai/Mistral-Nemo-Instruct-2407"
    MISTRAL_NEMO_MINITRON_8B_BASE = "nvidia/Mistral-NeMo-Minitron-8B-Base"
    META_LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
    META_LLAMA_3_8B_INSTRUCT = "meta-llama/Meta-Llama-3-8B-Instruct"
    META_LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    GEMMA_7B = "google/gemma-7b"
    GEMMA_11_7B_IT = "google/gemma-1.1-7b-it"
    GEMMA_2B = "google/gemma-2b"
    GEMMA_11_2B_IT = "google/gemma-1.1-2b-it"
    PHI_3_MINI_4K_INSTRUCT = "microsoft/Phi-3-mini-4k-instruct"
    FALCON_MAMBA_7B = "tiiuae/falcon-mamba-7b"
    GLIDER = "PatronusAI/glider"


class LLMServingType(Enum):
    """
    Enum for LLM serving types

    **Values**

    * **VLLM**: VLLM serving
    * **HUGGINGFACE**: Hugging Face serving
    * **HUGGINGFACE_API**: Hugging Face API serving

    """

    VLLM = "vllm"
    HUGGINGFACE = "huggingface"
    HUGGINGFACE_API = "huggingface_api"


class TestStatus(Enum):
    """
    Enum for test status

    **Values**

    * **PASSED**: Test passed
    * **FAILED**: Test failed
    """

    PASSED = "PASSED"
    FAILED = "FAILED"
