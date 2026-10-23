"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the ChestCNN model,
plus a weakly-supervised bounding box derived from the heatmap (Zhou et al.,
2016 CAM-localization technique) -- NOT a separately trained object detector.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def _jet_colormap(value):
    v = max(0.0, min(1.0, value))
    r = np.clip(1.5 - abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * v - 1), 0, 1)
    return int(r * 255), int(g * 255), int(b * 255)


def get_bounding_box(cam, threshold_frac=0.5):
    if cam.max() <= 0:
        return None
    mask = cam >= (threshold_frac * cam.max())
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    return int(row_min), int(row_max), int(col_min), int(col_max)


def scale_box_to_image(box, cam_shape, image_size):
    row_min, row_max, col_min, col_max = box
    cam_h, cam_w = cam_shape
    img_w, img_h = image_size
    x_min = int(col_min / cam_w * img_w)
    x_max = int((col_max + 1) / cam_w * img_w)
    y_min = int(row_min / cam_h * img_h)
    y_max = int((row_max + 1) / cam_h * img_h)
    return x_min, y_min, x_max, y_max


def draw_box(image, box, color=(255, 255, 255), width=3):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle(box, outline=color, width=width)
    return img_copy


def overlay_heatmap(original_img, cam, alpha=0.45):
    size = original_img.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
    cam_arr = np.array(cam_img) / 255.0

    unique_vals, inverse = np.unique(cam_arr, return_inverse=True)
    color_lookup = np.array([_jet_colormap(v) for v in unique_vals], dtype=np.uint8)
    heatmap_rgb = color_lookup[inverse].reshape(cam_arr.shape[0], cam_arr.shape[1], 3)

    heatmap_img = Image.fromarray(heatmap_rgb).convert("RGB")
    base_rgb = original_img.convert("RGB")
    blended = Image.blend(base_rgb, heatmap_img, alpha=alpha)
    return blended
