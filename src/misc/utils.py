from typing import Union, List, Dict
import json
import string

from langchain_core.documents.base import Document

from src.misc.create_unique_id import create_unique_id_from_str


def remove_punctuations(s: str, exclude: Union[List[str], str, None] = None) -> str:
    """
    **Remove Punctuations from a String**

    This function removes punctuations from a given string, with optional exclusion of specific punctuations.

    :param s: Input string to remove punctuations from
    :type s: str
    :param exclude: Optional list or single punctuation to exclude from removal, defaults to None
    :type exclude: Union[List[str], str, None], optional
    :return: String with punctuations removed
    :rtype: str
    """
    punctuations = list(string.punctuation)
    if exclude:
        if isinstance(exclude, str) and len(exclude) == 1:
            punctuations.remove(exclude)
        else:
            for exclude_punc in exclude:
                punctuations.remove(exclude_punc)
    s = s.translate(str.maketrans("", "", "".join(punctuations)))
    return s


def merge_documents(documents: List[Document]) -> Document:
    merged_page_content = ""
    merged_metadata: Dict[str, str] = {}
    for doc in documents:
        merged_page_content += doc.page_content

        merged_metadata = {**merged_metadata, **doc.metadata}
    if "page" in merged_metadata.keys():
        merged_metadata["pages"] = merged_metadata.pop("page")
    merged_metadata["id"] = create_unique_id_from_str(merged_metadata["source"])

    merged_document = Document(
        page_content=merged_page_content, metadata=merged_metadata
    )
    return merged_document


def remove_leading_endlines(text: str) -> str:

    while text[0] == "\n":
        text = text[1:]
    return text


if __name__ == "__main__":
    s = "\nIn this research paper, the authors propose a new self-supervised pretraining approach called Contrastive Learning of Audio Representations (COLA) for learning general-purpose representations of sounds beyond speech. The motivation behind this study is that most work on learning representations of audio has focused on speech tasks, ignoring other audio tasks such as acoustic scene detection or animal vocalizations. Additionally, triplet-based objectives rely heavily on the mining of negative samples, and the quality of learned features can vary significantly with the sample generation scheme. COLA uses a simple contrastive learning framework that generates similar pairs by sampling segments from the same audio clip and dissimilar pairs by associating segments from different clips in the same batch. This approach allows for the consideration of a large number of negatives for each positive pair in the loss function and bypasses the need for a careful choice of negative examples. COLA is effective on several diverse downstream tasks, including speech, music, acoustic scenes, and animal sounds, and outperforms previous self-supervised methods on these tasks. Future work includes exploring other similarity measures, investigating the impact of various pretraining batch sizes, and exploring the use of COLA for audio-visual representation learning or other multimodal applications. The study also demonstrates the effectiveness of COLA on various downstream tasks and compares it to previous self-supervised methods, standard triplet loss, CBOW and SG, and temporal gap prediction models, and shows that COLA embeddings consistently outperform all these methods on various tasks. The study also investigates the role of the similarity measure in the quality of learned representations and performs an ablation study to compare model pretraining with cosine and bilinear similarity, observing that the best results were obtained using bilinear similarity in all cases. Another experiment measures the impact of pretraining batch size on downstream test accuracy, showing"
    print(s[1])
