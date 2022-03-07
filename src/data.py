from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class InputExample:
    text: str
    text_pair: str
    label: str


@dataclass
class InputFeature:
    input_ids: List[int]
    token_type_ids: List[int]

    label: int