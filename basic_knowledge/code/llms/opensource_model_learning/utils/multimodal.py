"""Tokenizer-independent multimodal preprocessing for the learning models."""

import torch
from torch.nn import functional as F


class MultimodalProcessor:
    """Prepare image tensors and validate visual placeholder tokens."""

    def __init__(self, image_token_id: int, patch_size: int, image_size: int | None = None) -> None:
        """Store visual-token and resize settings.

        Parameters:
            image_token_id: Token ID reserved for one visual patch embedding.
            patch_size: Vision patch size used to calculate output token counts.
            image_size: Optional square target resolution before patchification.
        """
        self.image_token_id = image_token_id
        self.patch_size = patch_size
        self.image_size = image_size

    def __call__(self, input_ids: torch.Tensor, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalize images to BCHW float tensors and validate placeholder count.

        Parameters:
            input_ids: Token IDs containing image placeholder positions.
            images: Image tensor in ``CHW``, ``HWC``, ``BCHW``, or ``BHWC`` layout.
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("images must have three or four dimensions")
        if images.shape[-1] in {1, 3} and images.shape[1] not in {1, 3}:
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] not in {1, 3}:
            raise ValueError("images must have one or three channels")
        pixel_values = images.float()
        if images.dtype == torch.uint8:
            pixel_values = pixel_values / 255.0
        if self.image_size is not None:
            pixel_values = F.interpolate(pixel_values, size = (self.image_size, self.image_size), mode = "bilinear", align_corners = False)
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image height and width must be divisible by patch_size")
        expected_tokens = pixel_values.shape[0] * (height // self.patch_size) * (width // self.patch_size)
        actual_tokens = int((input_ids == self.image_token_id).sum().item())
        if actual_tokens != expected_tokens:
            raise ValueError("image placeholder count must equal total visual patch count")
        return {"input_ids": input_ids, "pixel_values": pixel_values}
