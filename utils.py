from typing import Annotated


def add_or_reset(existing: list, update):
    if update == "RESET":
        return []
    else:
        return existing + update