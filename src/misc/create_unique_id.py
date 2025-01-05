"""
Misc for creating unique id
===========================

"""

import hashlib
from datetime import datetime
import uuid


def create_unique_user_id() -> str:
    """
    **create unique user id constructed from date timestamp and uuid**

    :return: unique id string
    :rtype: str
    """
    return f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{str(uuid.uuid4())}"


def create_unique_id_from_str(string: str) -> str:
    """
    Generates a unique id name
    refs:
    - md5: https://stackoverflow.com/questions/22974499/generate-id-from-string-in-python
    - sha3: https://stackoverflow.com/questions/47601592/safest-way-to-generate-a-unique-hash
    (- guid/uiid: https://stackoverflow.com/questions/534839/how-to-create-a-guid-uuid-in-python?noredirect=1&lq=1)
    """

    m = hashlib.md5()
    string = string.encode("utf-8")
    m.update(string)
    return str(int(m.hexdigest(), 16))


if __name__ == "__main__":
    print(create_unique_id_from_str("https://arxiv.org/pdf/2010.10915"))
