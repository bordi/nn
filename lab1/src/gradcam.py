from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DEFAULT_MEAN, DEFAULT_STD, GRADCAM_DIR, GradCAMConfig, ensure_directories, get_device
from .data import create_eval_loader
from .model import get_gradcam_target_layer, load_checkpoint_model
from .progress import TerminalProgressBar


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(DEFAULT_MEAN)
    std = np.array(DEFAULT_STD)
    image = (image * std) + mean
    image = np.clip(image, 0.0, 1.0)
    return image


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if cam.shape[:2] != image.shape[:2]:
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam = (
            torch.nn.functional.interpolate(
                cam_tensor,
                size=image.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    heatmap = plt.cm.jet(cam)[..., :3]
    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0.0, 1.0)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def save_gradcam_figure(
    output_path: Path,
    original_image: np.ndarray,
    overlay_image: np.ndarray,
    true_label: str,
    predicted_label: str,
    is_correct: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(overlay_image)
    axes[1].set_title("Grad-CAM overlay")
    axes[1].axis("off")

    status = "correct" if is_correct else "incorrect"
    figure.suptitle(f"{status} | true={true_label} | predicted={predicted_label}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_gradcam(args) -> list[str]:
    ensure_directories()
    device = get_device(prefer_cpu=args.device == "cpu")
    test_loader, class_names = create_eval_loader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        debug_samples=args.debug_samples,
    )
    model, checkpoint = load_checkpoint_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(class_names),
        map_location="cpu",
    )
    model.to(device)
    model.eval()

    gradcam = GradCAM(model=model, target_layer=get_gradcam_target_layer(model))
    correct_target = args.correct_examples
    incorrect_target = args.incorrect_examples
    correct_found = 0
    incorrect_found = 0
    saved_paths: list[str] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = TerminalProgressBar(total=len(test_loader), description="gradcam")

    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        label_names = batch["label_name"]

        for item_index in range(images.size(0)):
            if correct_found >= correct_target and incorrect_found >= incorrect_target:
                gradcam.close()
                progress_bar.close(
                    postfix=f"correct={correct_found}/{correct_target} incorrect={incorrect_found}/{incorrect_target}"
                )
                print(f"Grad-CAM images saved to: {output_dir}")
                return saved_paths

            image_tensor = images[item_index : item_index + 1]
            target_index = int(labels[item_index].item())

            with torch.no_grad():
                logits = model(image_tensor)
                predicted_index = int(logits.argmax(dim=1).item())

            is_correct = predicted_index == target_index
            if is_correct and correct_found >= correct_target:
                continue
            if not is_correct and incorrect_found >= incorrect_target:
                continue

            cam = gradcam.generate(image_tensor, class_index=predicted_index)
            original_image = denormalize_image(image_tensor[0])
            overlay_image = overlay_heatmap(original_image, cam)

            status = "correct" if is_correct else "incorrect"
            counter = correct_found + 1 if is_correct else incorrect_found + 1
            output_path = output_dir / f"{status}_{counter:02d}.png"
            save_gradcam_figure(
                output_path=output_path,
                original_image=original_image,
                overlay_image=overlay_image,
                true_label=label_names[item_index],
                predicted_label=class_names[predicted_index],
                is_correct=is_correct,
            )
            saved_paths.append(str(output_path))

            if is_correct:
                correct_found += 1
            else:
                incorrect_found += 1

        progress_bar.update(
            postfix=f"correct={correct_found}/{correct_target} incorrect={incorrect_found}/{incorrect_target}"
        )

    gradcam.close()
    progress_bar.close(
        postfix=f"correct={correct_found}/{correct_target} incorrect={incorrect_found}/{incorrect_target}"
    )
    print(
        "Reached the end of the dataset before collecting all requested samples. "
        f"Saved {len(saved_paths)} images."
    )
    return saved_paths


def build_gradcam_defaults() -> dict:
    config = GradCAMConfig()
    return {
        "output_dir": str(GRADCAM_DIR),
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "correct_examples": config.correct_examples,
        "incorrect_examples": config.incorrect_examples,
        "debug_samples": config.debug_samples,
    }
