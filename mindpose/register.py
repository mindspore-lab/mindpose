"""Model registry
"""
import logging
from collections import defaultdict
from typing import Any, Callable, List

_module_to_models = defaultdict(set)
_model_to_module = dict()
_model_entrypoint = defaultdict(dict)


def _global_register(module_name: str, func_name: str, fn: Callable[..., Any]) -> None:
    if func_name in _model_entrypoint[module_name]:
        logging.warning(f"`{func_name}` is already registered")
    _model_entrypoint[module_name][func_name] = fn
    _model_to_module[func_name] = module_name
    _module_to_models[module_name].add(func_name)


def register(module_name: str, extra_name: str = "") -> Callable[..., Any]:
    """
    Register the function globally with function.__name__. If extra_name is provided,
    the function is registered twice with the given name.

    Args:
        module_name: the module name want to register
        extra_name: extra func name to register. Default: ""
    """

    def wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
        _global_register(module_name, fn.__name__, fn)
        if extra_name:
            _global_register(module_name, extra_name, fn)
        return fn

    return wrapper


def model_entrypoint(module_name: str, model_name: str) -> Callable[..., Any]:
    return _model_entrypoint[module_name][model_name]


def list_models(module: str) -> List[str]:
    models = sorted(list(_module_to_models[module]))
    return models


def list_modules() -> List[str]:
    modules = _module_to_models.keys()
    return sorted(list(modules))
