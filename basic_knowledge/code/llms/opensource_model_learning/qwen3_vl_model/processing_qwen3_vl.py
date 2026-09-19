# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-VL processor port without the Transformers processor runtime."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

sys.path.append(os.getcwd())

try:
    from ..utils.processing import MultiModalData, ProcessorMixin
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.processing import MultiModalData, ProcessorMixin


class Qwen3VLProcessor(ProcessorMixin):
    """Construct Qwen3-VL text placeholders and retain multimodal tensors.

    Parameters:
        image_processor: Qwen-compatible image processor exposing ``merge_size``.
        tokenizer: Tokenizer exposing special tokens and ``batch_decode``.
        video_processor: Qwen-compatible video processor exposing grid metadata.
        chat_template: Optional prompt template.
    """

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        video_processor: Any = None,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize official Qwen3-VL special-token attributes.

        Parameters:
            image_processor: Image processor used to calculate image patch grids.
            tokenizer: Text tokenizer owning Qwen special-token IDs.
            video_processor: Video processor used to calculate video patch grids.
            chat_template: Optional chat template.
            **kwargs: Reserved for source-compatible construction.
        """
        del kwargs
        if tokenizer is None:
            raise ValueError("Qwen3VLProcessor requires a tokenizer.")

        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template = chat_template)
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def replace_image_token(self, image_inputs: dict[str, Any], image_idx: int, **kwargs: Any) -> str:
        """Expand one Qwen image placeholder into merged visual-token placeholders.

        Parameters:
            image_inputs: Image processor output containing ``image_grid_thw``.
            image_idx: Index of the image being inserted into the prompt.
            **kwargs: Retained for the upstream processor callback contract.
        """
        del kwargs
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = image_inputs["image_grid_thw"][image_idx].prod() // merge_length
        return self.image_token * int(num_image_tokens)

    def replace_video_token(self, video_inputs: dict[str, Any], video_idx: int, **kwargs: Any) -> str:
        """Expand one Qwen video placeholder with timestamped frame token groups.

        Parameters:
            video_inputs: Video processor output containing grid and metadata fields.
            video_idx: Index of the video being inserted into the prompt.
            **kwargs: Retained for the upstream processor callback contract.
        """
        del kwargs
        merge_length = self.video_processor.merge_size**2
        num_frames = video_inputs["video_grid_thw"][video_idx][0]
        frame_seqlen = video_inputs["video_grid_thw"][video_idx][1:].prod() // merge_length
        metadata = video_inputs["video_metadata"][video_idx]
        video_placeholder = ""

        if metadata.fps is None:
            # The official processor defaults to 24 fps when metadata is unavailable.
            metadata.fps = 24

        # if timestamps are not provided, calculate them
        curr_timestamp = self._calculate_timestamps(
            metadata.frames_indices,
            metadata.fps,
            self.video_processor.temporal_patch_size,
        )

        for frame_idx in range(int(num_frames)):
            curr_time = curr_timestamp[frame_idx]
            video_placeholder += f"<{curr_time:.1f} seconds>"
            video_placeholder += self.vision_start_token + self.video_token * int(frame_seqlen) + self.vision_end_token
        return video_placeholder

    def _get_num_multimodal_tokens(
        self,
        image_sizes: list[list[int]] | None = None,
        video_sizes: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> MultiModalData:
        """Compute the number of official merged placeholders for image and video sizes.

        Parameters:
            image_sizes: ``(height, width)`` values, one for each image.
            video_sizes: ``(num_frames, height, width)`` values, one per video.
            **kwargs: Image/video processor overrides.
        """
        vision_data: dict[str, list[int]] = {}
        if image_sizes is not None:
            images_kwargs = dict(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [num_patches // merge_size**2 for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = dict(kwargs)
            merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            vision_data["num_video_tokens"] = [num_patches // merge_size**2 for num_patches in num_video_patches]

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs: Any,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Decode generated Qwen3-VL text with the supplied tokenizer.

        Parameters:
            generated_outputs: Token IDs returned by model generation.
            skip_special_tokens: Whether tokenizer special tokens are removed.
            clean_up_tokenization_spaces: Whether tokenizer whitespace cleanup is enabled.
            **kwargs: Extra ``batch_decode`` arguments.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens = skip_special_tokens,
            clean_up_tokenization_spaces = clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> list[str]:
        """Return component inputs plus the official multimodal token type IDs.

        Parameters:
            None.
        """
        return super().model_input_names + ["mm_token_type_ids"]

    def _calculate_timestamps(
        self,
        indices: list[int] | np.ndarray,
        video_fps: float,
        merge_size: int = 2,
    ) -> list[float]:
        """Return temporal-patch midpoint timestamps using the official algorithm.

        Parameters:
            indices: Frame indices produced by the video processor.
            video_fps: Frames per second of the source video.
            merge_size: Number of frames merged into one temporal patch.
        """
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # @JJJYmmm frames are merged by self.merge_size, \
        # so we need to average the timestamps between the first/last frame within the temporal patch
        return [
            (timestamps[index] + timestamps[index + merge_size - 1]) / 2
            for index in range(0, len(timestamps), merge_size)
        ]


__all__ = ["Qwen3VLProcessor"]
