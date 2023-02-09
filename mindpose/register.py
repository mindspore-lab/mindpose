"""Components registry
"""
import logging
from collections import defaultdict
from typing import Any, Callable, List

_module_to_components = defaultdict(set)
_components_to_module = dict()
_entrypoints = defaultdict(dict)

_logger = logging.getLogger(__name__)


def _global_register(module_name: str, func_name: str, fn: Callable[..., Any]) -> None:
    if func_name in _entrypoints[module_name]:
        _logger.warning(f"`{func_name}` is already registered")
    _entrypoints[module_name][func_name] = fn
    _components_to_module[func_name] = module_name
    _module_to_components[module_name].add(func_name)


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


def list_components(module: str) -> List[str]:
    components = sorted(list(_module_to_components[module]))
    return components


def list_modules() -> List[str]:
    modules = _module_to_components.keys()
    return sorted(list(modules))


def entrypoint(module_name: str, component_name: str) -> Callable[..., Any]:
    if module_name not in _entrypoints:
        raise ValueError(
            f"Unkown module `{module_name}`. " f"Supported modules: {list_modules()}"
        )
    if component_name not in _entrypoints[module_name]:
        raise ValueError(
            f"Unkown components `{component_name}`. "
            f"Supported componetns in `{module_name}`: {list_components(module_name)}"
        )
    return _entrypoints[module_name][component_name]
