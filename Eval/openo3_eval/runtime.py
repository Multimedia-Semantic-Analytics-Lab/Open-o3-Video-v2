"""Local logging, path, and plugin-loading helpers."""

import hashlib
import importlib.util
import logging
import os
import sys
from typing import Any


def get_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s:%(name)s] %(message)s")
    return logging.getLogger("openo3_eval")


def to_abspath(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def import_external_file(path: str) -> Any:
    file_path = to_abspath(path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"[Open-o3 Eval] external plugin file not found: {file_path}")
    module_name = f"openo3_eval_external_{hashlib.md5(file_path.encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[Open-o3 Eval] failed to load external plugin file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
