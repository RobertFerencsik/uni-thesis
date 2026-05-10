import json
from dataclasses import dataclass

from .paths import CONFIG_FILE


with CONFIG_FILE.open(encoding="utf-8") as f:
    _regexes_config = json.load(f)["regexes"]


@dataclass(frozen=True)
class Regexes:
    url: str
    email: str
    phone: str
    num: str
    uppercase: str
    repeated_char: str


REGEXES = Regexes(**_regexes_config)
