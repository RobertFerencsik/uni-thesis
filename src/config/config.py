try:
    from src.infrastructure.paths import PATHS, PROJECT_ROOT, Paths
    from src.infrastructure.regexes import REGEXES, Regexes
except ImportError:
    from infrastructure.paths import PATHS, PROJECT_ROOT, Paths
    from infrastructure.regexes import REGEXES, Regexes

__all__ = [
    "PATHS",
    "PROJECT_ROOT",
    "Paths",
    "REGEXES",
    "Regexes",
]
