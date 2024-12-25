from typing import Union, List, Dict
import json
import string


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
