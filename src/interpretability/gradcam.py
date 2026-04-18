from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    overlay_rgb: np.ndarray
    image_rgb: np.ndarray


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = None

    def _tensor_grad_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def _forward_hook(self, module: nn.Module, inputs, output) -> None:
        self.activations = output.detach()


        output.register_hook(self._tensor_grad_hook)

    def remove_hooks(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if logits.ndim == 2 and logits.shape[0] == 1:
            if class_idx is None:
                if logits.shape[1] == 1:
                    score = logits[0, 0]
                else:
                    raise ValueError("class_idx must be provided for multi-class/multi-label outputs.")
            else:
                score = logits[0, class_idx]
        else:
            raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if np.allclose(cam.max(), cam.min()):
            return np.zeros_like(cam, dtype=np.float32)

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)


def denormalize_image(
    image_tensor: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    img = img * std_arr + mean_arr
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
    return overlay.astype(np.uint8)


def run_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
    alpha: float = 0.35,
) -> GradCAMResult:
    cam = GradCAM(model=model, target_layer=target_layer)
    try:
        heatmap = cam(input_tensor=input_tensor, class_idx=class_idx)
    finally:
        cam.remove_hooks()

    image_rgb = denormalize_image(original_tensor[0])
    overlay_rgb = overlay_heatmap_on_image(image_rgb=image_rgb, heatmap=heatmap, alpha=alpha)

    return GradCAMResult(
        heatmap=heatmap,
        overlay_rgb=overlay_rgb,
        image_rgb=image_rgb,
    )