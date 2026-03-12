from pathlib import Path
import json
from dataclasses import dataclass
from sys import path_hooks

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# CONFIG_FILE = PROJECT_ROOT / "src" / "config" / "config.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "notebooks" / "config.json"

with CONFIG_FILE.open(encoding="utf-8") as f:
    _raw_config = json.load(f)
_paths_config = _raw_config["paths"]
_regexes_config = _raw_config["regexes"]

def _resolve_dir(key, value):
    return {key: PROJECT_ROOT / value}

def _resolve_file(value1, key, value2):
    return {key: value1 / value2}

def init_paths(config):
    resolved = {}

    for item in config:
        items = list(item.items())

        dir_key, dir_value = items[0]
        resolved.update(_resolve_dir(dir_key, dir_value))
        dir_path = resolved[dir_key]

        for file_key, filename in items[1:]:
            resolved.update(_resolve_file(dir_path, file_key, filename))

    return resolved

@dataclass(frozen = True)
class Paths:
    corpora_raw: Path
    corpus_raw: Path
    train_raw: Path
    validation_raw: Path
    test_raw: Path

    corpora_processed: Path
    train_processed: Path
    validation_processed: Path
    test_processed: Path
    lstm_tokenizer_train_data: Path

    models: Path
    lstm_tokenizer: Path

    def make_dirs(self):
        for path in self.__dict__.values():
            if path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)

PATHS = Paths(**init_paths(_paths_config))

@dataclass(frozen = True)
class Regexes:
    url: str
    email: str
    phone: str
    num: str
    uppercase: str
    repeated_char:str

REGEXES = Regexes(**_regexes_config)
