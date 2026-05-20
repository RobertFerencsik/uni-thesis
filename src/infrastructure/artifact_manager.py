from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch


class ArtifactManager:

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None

    def _resolved(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if self.base_dir is not None and not p.is_absolute():
            return (self.base_dir / p).resolve()
        return p.resolve()

    def make_dir(self, path: Union[str, Path]) -> Path:
        resolved = self._resolved(path)
        if resolved.suffix:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        else:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def save_torch(self, obj: Any, path: Union[str, Path]) -> None:
        out = self._resolved(path)
        self.make_dir(out.parent)
        torch.save(obj, out)

    def load_torch(
        self,
        path: Union[str, Path],
        *,
        map_location: Any = None,
    ) -> Any:
        return torch.load(self._resolved(path), map_location=map_location)

    def save_json(
        self,
        data: Any,
        path: Union[str, Path],
        *,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False,
    ) -> None:
        out = self._resolved(path)
        self.make_dir(out.parent)
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    def load_json(self, path: Union[str, Path]) -> Any:
        with self._resolved(path).open(encoding="utf-8") as f:
            return json.load(f)

    def load_csv(self, path: Union[str, Path], **kwargs: Any) -> Any:
        return pd.read_csv(self._resolved(path), **kwargs)

    def save_csv(
        self,
        dataframe: Any,
        path: Union[str, Path],
        **kwargs: Any,
    ) -> None:
        out = self._resolved(path)
        self.make_dir(out.parent)
        dataframe.to_csv(out, **kwargs)

    def save_figure(self, fig: Any, path: Union[str, Path], **kwargs: Any) -> None:
        out = self._resolved(path)
        self.make_dir(out.parent)
        fig.savefig(out, **kwargs)

    def write_text(
        self,
        path: Union[str, Path],
        content: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        out = self._resolved(path)
        self.make_dir(out.parent)
        out.write_text(content, encoding=encoding)
