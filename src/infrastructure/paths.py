import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "src" / "config" / "config.json"

with CONFIG_FILE.open(encoding="utf-8") as f:
    _raw_config = json.load(f)
_paths_config = _raw_config["paths"]


def _resolve_dir(key: str, value: str) -> dict:
    return {key: PROJECT_ROOT / value}


def _resolve_file(value1: Path, key: str, value2: str) -> dict:
    return {key: value1 / value2}


def init_paths(config: list) -> dict:
    resolved: dict = {}

    for item in config:
        items = list(item.items())

        dir_key, dir_value = items[0]
        resolved.update(_resolve_dir(dir_key, dir_value))
        dir_path = resolved[dir_key]

        for file_key, filename in items[1:]:
            resolved.update(_resolve_file(dir_path, file_key, filename))

    return resolved


@dataclass(frozen=True)
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
    train_ids: Path
    train_pieces: Path
    validation_ids: Path
    validation_pieces: Path
    test_ids: Path
    test_pieces: Path

    def make_dirs(self) -> None:
        for path in self.__dict__.values():
            if path.suffix == "":
                path.mkdir(parents=True, exist_ok=True)


PATHS = Paths(**init_paths(_paths_config))
