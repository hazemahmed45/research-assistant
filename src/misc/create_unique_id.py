"""
Misc for creating unique id
===========================

"""

import uuid
from datetime import datetime


def create_unique_user_id() -> str:
    """
    **create unique user id constructed from date timestamp and uuid**

    :return: unique id string
    :rtype: str
    """
    return f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{str(uuid.uuid4())}"
