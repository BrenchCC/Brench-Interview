"""Small configuration primitives shared by the local architecture ports.

This module deliberately replaces only the ``PreTrainedConfig`` lifecycle used
by the copied model configurations. It is not a replacement for the complete
Transformers configuration system.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class PreTrainedConfig:
    """Store declared configuration fields and run source-compatible post-init logic.

    Parameters:
        **kwargs: Values overriding declared model configuration fields. Extra
            values are retained because official checkpoints sometimes include
            backwards-compatible configuration keys.
    """

    model_type = ""
    attribute_map: dict[str, str] = {}

    def __init__(self, **kwargs: Any) -> None:
        """Copy class defaults, apply user values, and invoke ``__post_init__``.

        Parameters:
            **kwargs: Values supplied by a model config, toy config, or checkpoint.
        """
        aliases = type(self).attribute_map
        for config_class in reversed(type(self).mro()):
            annotations = getattr(config_class, "__annotations__", {})
            for name in annotations:
                if name != "attribute_map" and name in config_class.__dict__:
                    setattr(self, name, deepcopy(getattr(config_class, name)))

        for name, value in kwargs.items():
            setattr(self, aliases.get(name, name), value)
        self.__post_init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Resolve an official attribute alias when no concrete field exists.

        Parameters:
            name: Requested attribute name.
        """
        if name in self.attribute_map:
            return getattr(self, self.attribute_map[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __post_init__(self, **kwargs: Any) -> None:
        """Provide the base post-init hook used by official configuration classes.

        Parameters:
            **kwargs: Retained for source-compatible subclass calls.
        """
        del kwargs

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copied serializable view of this configuration.

        Parameters:
            None.
        """
        result = deepcopy(self.__dict__)
        result["model_type"] = self.model_type
        return result


def remap_legacy_layer_types(layer_types: list[str]) -> list[str]:
    """Map legacy configuration layer-type names to current source names.

    Parameters:
        layer_types: Layer schedule read from a configuration or checkpoint.
    """
    mapping = {
        "attention": "full_attention",
        "linear": "linear_attention",
        "full": "full_attention",
    }
    return [mapping.get(layer_type, layer_type) for layer_type in layer_types]
