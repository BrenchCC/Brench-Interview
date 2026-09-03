"""Minimal processor primitives needed by the copied multimodal processors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MultiModalData:
    """Hold the multimodal placeholder counts calculated by a processor.

    Parameters:
        num_image_tokens: Number of merged image tokens per input image.
        num_image_patches: Number of raw image patches per input image.
        num_video_tokens: Number of merged video tokens per input video.
    """

    num_image_tokens: list[int] | None = None
    num_image_patches: list[int] | None = None
    num_video_tokens: list[int] | None = None


class ProcessorMixin:
    """Keep the component ownership and input-name contract of HF processors.

    Parameters:
        image_processor: Callable image preprocessor.
        tokenizer: Text tokenizer implementing ``__call__`` and ``batch_decode``.
        video_processor: Callable video preprocessor.
        chat_template: Optional prompt template retained as metadata.
    """

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        video_processor: Any = None,
        chat_template: str | None = None,
    ) -> None:
        """Store the component processors without importing Transformers.

        Parameters:
            image_processor: Callable image preprocessor.
            tokenizer: Text tokenizer.
            video_processor: Callable video preprocessor.
            chat_template: Optional chat template.
        """
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.chat_template = chat_template

    @property
    def model_input_names(self) -> list[str]:
        """Return the union of component input names.

        Parameters:
            None.
        """
        names: list[str] = []
        for component in (self.tokenizer, self.image_processor, self.video_processor):
            for name in getattr(component, "model_input_names", []):
                if name not in names:
                    names.append(name)
        return names
