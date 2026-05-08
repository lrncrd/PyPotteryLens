# utils.py

import numpy as np
import os
import json
import logging
import threading
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
from PIL import Image
import fitz
from skimage.filters import threshold_otsu, median
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, disk
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import shutil
from reportlab.lib import pagesizes
from reportlab.pdfgen import canvas
import gc

logger = logging.getLogger(__name__)



@dataclass
class PDFConfig:
    """Configuration for PDF processing"""
    output_dir: Path

@dataclass
class FewShotConfig:
    """Configuration for SAM2 + DINOv2 few-shot processor"""
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    sam2_checkpoint_url: str = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    sam2_checkpoint_path: Path = Path("models_vision/sam2.1_hiera_large.pt")
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    dinov2_repo: str = "facebookresearch/dinov2"
    dinov2_model: str = "dinov2_vitl14"
    # Local directory for torch.hub cache (DINOv2 weights stored here)
    dinov2_hub_dir: Path = Path("models_vision/hub")
    # Pre-downloaded weights file (placed by initialize_models at startup)
    dinov2_weights_url: str = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
    # SAM2 automatic mask generator parameters
    points_per_side: int = 16          # 16²=256 grid pts vs 32²=1024 → ~4× faster
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 200    # discard tiny noise masks

@dataclass
class MaskExtractionConfig:
    """Configuration for mask extraction"""
    pdfimg_output_dir: Path  # Directory containing the original images
    pred_output_dir: Path    # Directory for predictions and output
    min_area_ratio: float = 0.001 # 0.005
    closing_kernel_size: int = 3
    output_suffix: str = "_card"
    mask_suffix: str = "_mask"

@dataclass
class TabularConfig:
    """Configuration for tabular processing"""
    pdfimg_output_dir: Path
    pred_output_dir: Path
    max_workers: int = 4  # For parallel processing
    cache_size: int = 32  # For LRU cache

class PDFProcessor:
    """Handles PDF to image conversion using PyMuPDF"""
    
    def __init__(self, config):
        self.config = config

    def process_pdf(self, pdf_path: str, split_pages: bool = False) -> str:
        """
        Convert PDF to images with optional page splitting
        
        Args:
            pdf_path: Path to PDF file
            split_pages: If True, splits each page into left and right halves
        """
        try:
            pdf_file_name = Path(pdf_path).stem
            output_folder = self.config.output_dir / pdf_file_name
            os.makedirs(output_folder, exist_ok=True)
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get the pixel map with a good resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))

                
                # Convert to PIL Image
                img_data = pix.samples
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                if split_pages:
                    self._process_split_page(img, pdf_file_name, page_num, output_folder)
                else:
                    self._process_single_page(img, pdf_file_name, page_num, output_folder)
            
            doc.close()
            return f"PDF file {pdf_file_name} has been converted to JPG"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def process_pdf_to_folder(self, pdf_path: str, output_folder: str, split_pages: bool = False, project_name: str = None) -> str:
        """
        Convert PDF to images with optional page splitting, saving to specified folder
        
        Args:
            pdf_path: Path to PDF file
            output_folder: Destination folder for images
            split_pages: If True, splits each page into left and right halves
            project_name: Optional project name to use for image naming (defaults to PDF filename)
        """
        try:
            # Use project name if provided, otherwise fall back to PDF filename
            base_name = project_name if project_name else Path(pdf_path).stem
            output_path = Path(output_folder)
            os.makedirs(output_path, exist_ok=True)
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get the pixel map with a good resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert to PIL Image
                img_data = pix.samples
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                if split_pages:
                    self._process_split_page(img, base_name, page_num, output_path)
                else:
                    self._process_single_page(img, base_name, page_num, output_path)
            
            doc.close()
            return f"PDF file {base_name} has been converted to JPG"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def _process_single_page(self, image: Image.Image, pdf_name: str, page_num: int, output_folder: Path):
        """Save a single page as one image"""
        output_image_name = f'{pdf_name}_page_{page_num}.jpg'
        image.save(output_folder / output_image_name, 'JPEG')

    def _process_split_page(self, image: Image.Image, pdf_name: str, page_num: int, output_folder: Path):
        """Split a page into left and right halves and save separately"""
        width, height = image.size
        mid_point = width // 2

        # Split into left and right pages
        left_page = image.crop((0, 0, mid_point, height))
        right_page = image.crop((mid_point, 0, width, height))

        # Save both pages with appropriate numbering
        left_page.save(output_folder / f'{pdf_name}_page_{page_num}a.jpg', 'JPEG')
        right_page.save(output_folder / f'{pdf_name}_page_{page_num}b.jpg', 'JPEG')

        
# ─────────────────────────────────────────────────────────────────────────────
# SAM2 + DINOv2 singletons (lazy-loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────
_sam2_predictor = None
_sam2_generator = None
_dinov2_model = None
_models_lock = threading.Lock()

_DINOV2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _ensure_sam2_checkpoint(config: FewShotConfig) -> None:
    """Download SAM2 checkpoint if not present."""
    ckpt = Path(config.sam2_checkpoint_path)
    if ckpt.exists():
        return
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading SAM2 checkpoint → {ckpt} …")
    urllib.request.urlretrieve(config.sam2_checkpoint_url, str(ckpt))
    logger.info("SAM2 checkpoint downloaded.")


def get_sam2_predictor(config: FewShotConfig):
    global _sam2_predictor
    with _models_lock:
        if _sam2_predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            _ensure_sam2_checkpoint(config)
            device = torch.device(config.device)
            sam2 = build_sam2(config.sam2_config, str(config.sam2_checkpoint_path), device=device)
            _sam2_predictor = SAM2ImagePredictor(sam2)
            logger.info("SAM2 predictor ready.")
    return _sam2_predictor


def get_sam2_generator(config: FewShotConfig):
    global _sam2_generator
    with _models_lock:
        if _sam2_generator is None:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            _ensure_sam2_checkpoint(config)
            device = torch.device(config.device)
            sam2 = build_sam2(config.sam2_config, str(config.sam2_checkpoint_path), device=device)
            _sam2_generator = SAM2AutomaticMaskGenerator(
                model=sam2,
                points_per_side=config.points_per_side,
                pred_iou_thresh=config.pred_iou_thresh,
                stability_score_thresh=config.stability_score_thresh,
                min_mask_region_area=config.min_mask_region_area,
            )
            logger.info("SAM2 generator ready.")
    return _sam2_generator


def get_dinov2(config: FewShotConfig):
    global _dinov2_model
    with _models_lock:
        if _dinov2_model is None:
            hub_dir = Path(config.dinov2_hub_dir)
            hub_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(hub_dir))
            logger.info(f"Loading DINOv2 ({config.dinov2_model}) from {hub_dir} …")
            _dinov2_model = torch.hub.load(
                config.dinov2_repo, config.dinov2_model,
                trust_repo=True,
            )
            _dinov2_model.to(torch.device(config.device))
            _dinov2_model.eval()
            logger.info("DINOv2 ready.")
    return _dinov2_model


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_mask_feature(image_rgb: np.ndarray, mask: np.ndarray, model: torch.nn.Module, device: str) -> np.ndarray:
    """Extract L2-normalised DINOv2 CLS embedding for a masked region (1024-dim)."""
    if mask.dtype != bool:
        mask = mask > 0
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros(1024, dtype=np.float32)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = image_rgb[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 0
    tensor = _DINOV2_TRANSFORM(Image.fromarray(crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
    feat = F.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy().flatten().astype(np.float32)


def _extract_mask_features_batch(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    model: torch.nn.Module,
    device: str,
    batch_size: int = 32,
) -> List[np.ndarray]:
    """
    Batch DINOv2 inference for a list of masks.
    Much faster than calling _extract_mask_feature one-by-one.
    Returns a list of (1024,) float32 vectors, one per input mask.
    """
    result = [np.zeros(1024, dtype=np.float32)] * len(masks)
    if not masks:
        return result

    tensors: List = []
    valid_idx: List[int] = []

    for i, mask in enumerate(masks):
        if mask.dtype != bool:
            mask = mask > 0
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = image_rgb[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 0
        tensors.append(_DINOV2_TRANSFORM(Image.fromarray(crop)))
        valid_idx.append(i)

    for batch_start in range(0, len(tensors), batch_size):
        batch_t = torch.stack(tensors[batch_start:batch_start + batch_size]).to(device)
        with torch.no_grad():
            feats = model(batch_t)                            # (B, 1024)
            feats = F.normalize(feats, p=2, dim=1)
        np_feats = feats.cpu().numpy().astype(np.float32)
        for j, orig_i in enumerate(valid_idx[batch_start:batch_start + batch_size]):
            result[orig_i] = np_feats[j]

    return result


def _mask_to_contour(mask: np.ndarray) -> List[List[int]]:
    """Simplify boolean mask to polygon contour [[x,y], ...]."""
    import cv2
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    eps = 0.003 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx.reshape(-1, 2).tolist()


def _compute_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


_CLASS_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#ec4899", "#14b8a6", "#f97316",
]


def _next_color(used: set) -> str:
    for c in _CLASS_COLORS:
        if c not in used:
            return c
    return _CLASS_COLORS[len(used) % len(_CLASS_COLORS)]


def _nms_masks(preds: List[Dict], all_masks: List[Dict], iou_threshold: float = 0.30) -> List[Dict]:
    """
    Intra-class NMS: given predictions sorted by similarity (best first),
    suppress any prediction whose mask IoU with a kept mask exceeds iou_threshold.
    Uses the actual boolean mask arrays for exact IoU.
    """
    kept: List[Dict] = []
    kept_masks: List[np.ndarray] = []
    for pred in preds:
        mask = all_masks[pred["index"]]["mask"]  # bool (H,W)
        suppress = False
        for km in kept_masks:
            inter = np.logical_and(mask, km).sum()
            union = np.logical_or(mask, km).sum()
            if union > 0 and inter / union >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(pred)
            kept_masks.append(mask)
    return kept


def _recompute_predictions(
    all_masks: List[Dict],
    classes: Dict,
    threshold: float,
    target_classes: Optional[Dict] = None,
) -> None:
    """
    Cross-class competition: each mask → best class (highest cosine similarity
    above threshold).
    - `classes`        : provides example features for similarity scoring (may span all images)
    - `target_classes` : receives the resulting predictions (defaults to `classes` if None)
    Applies intra-class NMS (IoU ≥ 0.30) to suppress redundant overlapping masks.
    """
    if target_classes is None:
        target_classes = classes
    # Ensure every class in `classes` has a slot in target_classes
    for cname, cdata in classes.items():
        if cname not in target_classes:
            target_classes[cname] = {
                "color": cdata["color"],
                "examples": [],
                "predictions": [],
                "rejected": set(),
            }
    # Exclude masks already labeled on this image (from target_classes, i.e. local)
    occupied = {ex["mask_index"] for c in target_classes.values() for ex in c["examples"] if ex.get("mask_index", -1) != -1}

    class_features: Dict[str, np.ndarray] = {}
    for cname, cdata in classes.items():
        feats = [ex["feature"] for ex in cdata["examples"]]
        if feats:
            class_features[cname] = np.array(feats)  # (M, 1024)

    best_assignment: Dict[str, list] = {cn: [] for cn in classes}

    for i, md in enumerate(all_masks):
        if i in occupied:
            continue
        mf = md["feature"]
        best_class, best_sim = None, -1.0
        for cname, fmat in class_features.items():
            rejected = classes[cname].get("rejected", set())
            if i in rejected:
                continue
            sims = np.dot(fmat, mf)          # (M,)
            sim = float(np.max(sims)) if len(sims) else -1.0
            if sim >= threshold and sim > best_sim:
                best_sim, best_class = sim, cname
        if best_class is not None:
            best_assignment[best_class].append({
                "index": i,
                "similarity": round(best_sim, 4),
                "contour": md["contour"],
                "area": md["area"],
                "bbox": md["bbox"],
            })

    for cname, preds in best_assignment.items():
        preds.sort(key=lambda p: p["similarity"], reverse=True)
        target_classes[cname]["predictions"] = _nms_masks(preds, all_masks, iou_threshold=0.30)


# ─────────────────────────────────────────────────────────────────────────────
# FewShotProcessor
# ─────────────────────────────────────────────────────────────────────────────

class FewShotProcessor:
    """
    SAM2 + DINOv2 few-shot detection engine.

    Workflow per project:
      1. process_image()        – run SAM2 automask + DINOv2 on one page; cache state
      2. add_example()          – user clicks a point → SAM2 segment → DINOv2 feature → class
      3. remove_example()       – remove a labelled example from a class
      4. get_predictions()      – return current per-class predictions for an image
      5. apply_to_all()         – batch: run SAM2+DINOv2 on all remaining pages,
                                   reuse class features → save mask PNG per image
      6. confirm_predictions()  – write confirmed mask indices as RGBA PNG to masks/

    State is kept in memory (keyed by image_id) AND persisted to
    <project_path>/fewshot/<image_id>/  as:
      - masks.npz     : per-mask bool arrays
      - features.npy  : (N, 1024) float32 features
      - meta.json     : contours, areas, bboxes
      - classes.json  : class names, colors, examples (features embedded)
    """

    def __init__(self, config: FewShotConfig):
        self.config = config
        self.device = config.device
        # In-memory store: {image_id: {image_np, all_masks, classes}}
        self._store: Dict[str, Dict[str, Any]] = {}

    # ── model accessors ───────────────────────────────────────────────────────

    @property
    def _predictor(self):
        return get_sam2_predictor(self.config)

    @property
    def _generator(self):
        return get_sam2_generator(self.config)

    @property
    def _dinov2(self):
        return get_dinov2(self.config)

    # ── public API ────────────────────────────────────────────────────────────

    def process_image(self, image_path: str, image_id: str, project_fewshot_dir: Path,
                       threshold: float = 0.5) -> Dict:
        """
        Run SAM2 automatic mask generation + DINOv2 feature extraction on one image.
        Result is cached in memory and persisted to disk.

        Returns dict with image_id, image_width, image_height, num_masks.
        """
        image_path = Path(image_path)
        image_np = np.array(Image.open(image_path).convert("RGB"))

        logger.info(f"[{image_id}] SAM2 automask on {image_np.shape} …")
        masks_data = self._generator.generate(image_np)
        logger.info(f"[{image_id}] {len(masks_data)} masks found.")

        # ── batch DINOv2 extraction (all masks at once) ───────────────────────
        raw_masks = [md["segmentation"] for md in masks_data]
        logger.info(f"[{image_id}] DINOv2 batch extraction for {len(raw_masks)} masks …")
        features = _extract_mask_features_batch(image_np, raw_masks, self._dinov2, self.device)

        all_masks = []
        for md, feature in zip(masks_data, features):
            all_masks.append({
                "mask": md["segmentation"],
                "feature": feature,
                "contour": _mask_to_contour(md["segmentation"]),
                "area": int(md["area"]),
                "bbox": [int(v) for v in md["bbox"]],
            })
        logger.info(f"[{image_id}] Feature extraction done.")

        # ── load per-image examples (for display/saving) ─────────────────────
        local_classes = self._load_classes(project_fewshot_dir, image_id)
        # ── aggregate ALL examples from disk (for feature matching only) ──────
        inference_classes = self._load_all_classes(project_fewshot_dir)
        # Ensure local_classes has a slot for every known class
        for cname, cdata in inference_classes.items():
            if cname not in local_classes:
                local_classes[cname] = {
                    "color": cdata["color"],
                    "examples": [],
                    "predictions": [],
                    "rejected": set(),
                }

        self._store[image_id] = {
            "image_np": image_np,
            "all_masks": all_masks,
            "classes": local_classes,            # per-image examples + predictions
            "inference_classes": inference_classes,  # all-image examples (inference only)
        }

        # Compute predictions using all features; store results in local_classes
        if inference_classes:
            _recompute_predictions(all_masks, inference_classes, threshold,
                                   target_classes=local_classes)
            logger.info(f"[{image_id}] Predictions recomputed (threshold={threshold}).")

        self._persist_state(project_fewshot_dir, image_id)

        return {
            "image_id": image_id,
            "image_width": int(image_np.shape[1]),
            "image_height": int(image_np.shape[0]),
            "num_masks": len(all_masks),
        }

    def load_for_prediction(self, image_path: str, image_id: str, project_fewshot_dir: Optional[Path] = None) -> Dict:
        """
        Load image into the store WITHOUT running SAM2 auto-mask generation.
        Fast (< 1 s) — just reads the image and makes it available for
        preview_prompt() and add_example().  Restores saved classes from disk
        if project_fewshot_dir is given.  DINOv2 features for 'apply_to_all'
        are computed lazily by process_image().

        Returns { image_id, image_width, image_height, classes }.
        """
        image_path = Path(image_path)
        if image_id in self._store:
            data = self._store[image_id]
            return {
                "image_id": image_id,
                "image_width": int(data["image_np"].shape[1]),
                "image_height": int(data["image_np"].shape[0]),
                "classes": self._classes_summary(data["classes"]),
            }
        image_np = np.array(Image.open(image_path).convert("RGB"))
        local_classes = self._load_classes(project_fewshot_dir, image_id) if project_fewshot_dir else {}
        inference_classes = self._load_all_classes(project_fewshot_dir) if project_fewshot_dir else {}
        for cname, cdata in inference_classes.items():
            if cname not in local_classes:
                local_classes[cname] = {
                    "color": cdata["color"],
                    "examples": [],
                    "predictions": [],
                    "rejected": set(),
                }
        self._store[image_id] = {
            "image_np": image_np,
            "all_masks": [],
            "classes": local_classes,
            "inference_classes": inference_classes,
        }
        return {
            "image_id": image_id,
            "image_width": int(image_np.shape[1]),
            "image_height": int(image_np.shape[0]),
            "classes": self._classes_summary(local_classes),
        }

    def add_example(
        self,
        image_id: str,
        class_name: str,
        x: int,
        y: int,
        threshold: float,
        project_fewshot_dir: Path,
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[List[int]] = None,    # [x1, y1, x2, y2]
    ) -> Dict:
        """
        User confirms a prompt (multi-point or box) to label an example for class_name.
        SAM2 segments it, DINOv2 extracts feature, added to class.
        Predictions for ALL classes are recomputed.
        Returns summary of all classes.
        """
        data = self._store.get(image_id)
        if data is None:
            raise ValueError(f"Image {image_id} not loaded. Call process_image first.")

        image_np = data["image_np"]
        all_masks = data["all_masks"]
        classes = data["classes"]

        # ── run SAM2 predictor ────────────────────────────────────────────────
        self._predictor.set_image(image_np)
        if box is not None:
            box_np = np.array(box, dtype=np.float32)  # [x1,y1,x2,y2]
            if points and labels:
                masks_out, scores, _ = self._predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array(labels),
                    box=box_np,
                    multimask_output=False,
                )
            else:
                masks_out, scores, _ = self._predictor.predict(
                    box=box_np,
                    multimask_output=False,
                )
        elif points and labels:
            masks_out, scores, _ = self._predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                multimask_output=False,
            )
        else:
            masks_out, scores, _ = self._predictor.predict(
                point_coords=np.array([[x, y]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )

        mask = masks_out[0]
        sam_score = float(scores[0])
        feature = _extract_mask_feature(image_np, mask, self._dinov2, self.device)

        # Match to precomputed mask by IoU
        best_iou, best_idx = 0.0, -1
        for i, md in enumerate(all_masks):
            iou = _compute_iou(mask, md["mask"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        mask_index = best_idx if best_iou > 0.8 else -1

        # Create class if new
        if class_name not in classes:
            used_colors = {c["color"] for c in classes.values()}
            classes[class_name] = {
                "color": _next_color(used_colors),
                "examples": [],
                "predictions": [],
                "rejected": set(),
            }

        new_example = {
            "id": len(classes[class_name]["examples"]),
            "mask_index": mask_index,
            "x": x,
            "y": y,
            "points": points or [[x, y]],
            "labels": labels or [1],
            "box": box,
            "mask": mask,
            "feature": feature,
            "contour": _mask_to_contour(mask),
            "area": int(mask.sum()),
            "sam_score": round(sam_score, 4),
        }
        classes[class_name].get("rejected", set()).clear()
        classes[class_name]["examples"].append(new_example)

        # Mirror the new example into inference_classes so predictions stay current
        inference_classes = data.get("inference_classes", classes)
        if class_name not in inference_classes:
            inference_classes[class_name] = {
                "color": classes[class_name]["color"],
                "examples": [],
                "predictions": [],
                "rejected": set(),
            }
        inference_classes[class_name]["examples"].append(new_example)

        _recompute_predictions(all_masks, inference_classes, threshold,
                               target_classes=classes)
        self._persist_state(project_fewshot_dir, image_id)

        return self._classes_summary(classes)

    def remove_example(
        self,
        image_id: str,
        class_name: str,
        example_index: int,
        threshold: float,
        project_fewshot_dir: Path,
    ) -> Dict:
        """Remove an example from a class and recompute predictions."""
        data = self._store.get(image_id)
        if data is None:
            raise ValueError(f"Image {image_id} not loaded.")
        classes = data["classes"]
        if class_name not in classes:
            raise ValueError(f"Class '{class_name}' not found.")
        cdata = classes[class_name]
        if example_index < 0 or example_index >= len(cdata["examples"]):
            raise ValueError("Invalid example index.")
        cdata["examples"].pop(example_index)
        # Re-number ids
        for i, ex in enumerate(cdata["examples"]):
            ex["id"] = i
        if not cdata["examples"]:
            del classes[class_name]
        # Reload inference_classes from disk to keep them consistent
        data["inference_classes"] = self._load_all_classes(project_fewshot_dir)
        # Mirror local new examples into inference_classes
        for cname2, cdata2 in classes.items():
            if cname2 not in data["inference_classes"]:
                data["inference_classes"][cname2] = {
                    "color": cdata2["color"], "examples": list(cdata2["examples"]),
                    "predictions": [], "rejected": set(),
                }
        inference_cls = data.get("inference_classes", classes)
        if inference_cls:
            _recompute_predictions(data["all_masks"], inference_cls, threshold,
                                   target_classes=classes)
        self._persist_state(project_fewshot_dir, image_id)
        return self._classes_summary(classes)

    def get_predictions(self, image_id: str, threshold: float) -> Dict:
        """Return current predictions for all classes on image_id (recomputes if needed)."""
        data = self._store.get(image_id)
        if data is None:
            raise ValueError(f"Image {image_id} not loaded.")
        inference_cls = data.get("inference_classes", data["classes"])
        _recompute_predictions(data["all_masks"], inference_cls, threshold,
                               target_classes=data["classes"])
        return self._classes_summary(data["classes"])

    def preview_prompt(
        self,
        image_id: str,
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        box: Optional[List[int]] = None,    # [x1, y1, x2, y2]
    ) -> Dict:
        """
        Run SAM2 predictor with the given point/box prompt and return the mask
        contour immediately, WITHOUT adding anything to a class.
        Used for live visual feedback while the user is building a prompt.

        Returns:
          { contour, area, bbox, sam_score, success }
        """
        data = self._store.get(image_id)
        if data is None:
            raise ValueError(f"Image {image_id} not loaded. Call process_image first.")

        image_np = data["image_np"]

        self._predictor.set_image(image_np)
        if box is not None:
            box_np = np.array(box, dtype=np.float32)
            if points and labels:
                masks_out, scores, _ = self._predictor.predict(
                    point_coords=np.array(points),
                    point_labels=np.array(labels),
                    box=box_np,
                    multimask_output=False,
                )
            else:
                masks_out, scores, _ = self._predictor.predict(
                    box=box_np,
                    multimask_output=False,
                )
        elif points and labels:
            masks_out, scores, _ = self._predictor.predict(
                point_coords=np.array(points),
                point_labels=np.array(labels),
                multimask_output=False,
            )
        else:
            return {"contour": [], "area": 0, "bbox": [], "sam_score": 0.0}

        mask = masks_out[0]
        sam_score = float(scores[0])
        contour = _mask_to_contour(mask)
        ys, xs = np.where(mask)
        bbox = [] if len(ys) == 0 else [
            int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min()), int(ys.max() - ys.min()),
        ]
        return {
            "contour": contour,
            "area": int(mask.sum()),
            "bbox": bbox,
            "sam_score": round(sam_score, 4),
        }

    def confirm_predictions(
        self,
        image_id: str,
        class_name: str,
        mask_indices: List[int],
        masks_output_dir: Path,
        image_filename: str,
    ) -> str:
        """
        Write confirmed mask indices as RGBA PNG (same format as old pipeline) to masks_output_dir.
        One file per original image: <stem>_mask_layer.png
        """
        data = self._store.get(image_id)
        if data is None:
            raise ValueError(f"Image {image_id} not loaded.")
        all_masks = data["all_masks"]
        image_np = data["image_np"]
        H, W = image_np.shape[:2]

        # Combine selected masks
        combined = np.zeros((H, W), dtype=bool)
        for idx in mask_indices:
            if 0 <= idx < len(all_masks):
                combined |= all_masks[idx]["mask"]

        _save_mask_rgba(combined, masks_output_dir, Path(image_filename).stem)
        return f"Saved {len(mask_indices)} mask(s) for '{class_name}' → {masks_output_dir}"

    def save_labeled_masks(
        self,
        image_id: str,
        masks_output_dir: Path,
        image_filename: str,
    ) -> Dict:
        """
        Save all labeled examples for every class on image_id as RGBA mask PNGs.
        Works both with and without auto-mask pre-processing.
        Each class whose examples have masks produces one combined PNG:
          <stem>_<class_name>_mask.png

        Returns { saved: {class_name: path_str, ...}, skipped: [class_name, ...] }
        """
        data = self._store.get(image_id)
        if data is None:
            raise ValueError(f"Image {image_id} not loaded.")

        image_np = data["image_np"]
        classes = data["classes"]
        H, W = image_np.shape[:2]
        masks_output_dir = Path(masks_output_dir)
        masks_output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_filename).stem

        saved, skipped = {}, []
        for class_name, cdata in classes.items():
            combined = np.zeros((H, W), dtype=bool)
            any_mask = False
            for ex in cdata["examples"]:
                ex_mask = ex.get("mask")
                if ex_mask is not None:
                    combined |= (ex_mask > 0) if not isinstance(ex_mask, np.ndarray) else ex_mask.astype(bool)
                    any_mask = True
                elif ex.get("mask_index", -1) >= 0:
                    idx = ex["mask_index"]
                    if idx < len(data["all_masks"]):
                        combined |= data["all_masks"][idx]["mask"]
                        any_mask = True
            if any_mask:
                safe_cls = "".join(c if c.isalnum() or c in "-_" else "_" for c in class_name)
                out_path = masks_output_dir / f"{stem}_{safe_cls}_mask.png"
                _save_mask_rgba(combined, masks_output_dir, f"{stem}_{safe_cls}")
                saved[class_name] = str(out_path)
            else:
                skipped.append(class_name)

        return {"saved": saved, "skipped": skipped}



    def apply_to_all(
        self,
        images_dir: Path,
        masks_dir: Path,
        project_fewshot_dir: Path,
        threshold: float,
        excluded_images: Optional[List[str]] = None,
        progress_callback=None,
    ) -> str:
        """
        Batch apply current class examples to every image in images_dir.
        For each image:
          - Run SAM2 automask + DINOv2
          - Cross-class competition with all class examples
          - Save combined mask per class as RGBA PNG in masks_dir

        progress_callback(current, total, message) is called if provided.
        """
        import re as _re

        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Collect examples from disk (all labeled images) + in-memory store
        merged_classes = self._load_all_classes(project_fewshot_dir)
        # Also merge anything only in memory (not yet flushed)
        for data in self._store.values():
            for cname, cdata in data["classes"].items():
                if cname not in merged_classes:
                    merged_classes[cname] = {
                        "color": cdata["color"], "examples": [],
                        "predictions": [], "rejected": set(),
                    }
                for ex in cdata["examples"]:
                    merged_classes[cname]["examples"].append(ex)

        all_class_features: Dict[str, np.ndarray] = {}
        all_class_colors: Dict[str, str] = {}
        for cname, cdata in merged_classes.items():
            feats = [ex["feature"] for ex in cdata["examples"]]
            if feats:
                all_class_features[cname] = np.array(feats)
                all_class_colors[cname]   = cdata["color"]

        if not all_class_features:
            return "No examples defined. Label at least one object first."

        ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        def _nat(s):
            return [int(c) if c.isdigit() else c.lower() for c in _re.split(r'(\d+)', s)]

        excluded_set = set(excluded_images or [])
        images = sorted(
            [f.name for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ext and f.name not in excluded_set],
            key=_nat,
        )

        total = len(images)
        if total == 0:
            return "No images found in project."

        for idx, img_name in enumerate(images, 1):
            if progress_callback:
                progress_callback(idx, total, f"Processing {img_name}")
            try:
                self._apply_to_single_image(
                    images_dir / img_name,
                    masks_dir,
                    all_class_features,
                    threshold,
                    project_fewshot_dir,
                )
            except Exception as e:
                logger.warning(f"Skipping {img_name}: {e}")

        return f"Batch complete: {total} images processed → {masks_dir}"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _apply_to_single_image(
        self,
        image_path: Path,
        masks_dir: Path,
        class_features: Dict[str, np.ndarray],
        threshold: float,
        project_fewshot_dir: Path,
    ) -> None:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        H, W = image_np.shape[:2]
        masks_data = self._generator.generate(image_np)

        raw_masks = [md["segmentation"] for md in masks_data]
        features  = _extract_mask_features_batch(image_np, raw_masks, self._dinov2, self.device)
        all_masks = [
            {"mask": md["segmentation"], "feature": feat, "area": int(md["area"])}
            for md, feat in zip(masks_data, features)
        ]

        # Cross-class competition
        combined: Dict[str, np.ndarray] = {cn: np.zeros((H, W), dtype=bool) for cn in class_features}

        for md in all_masks:
            mf = md["feature"]
            best_class, best_sim = None, -1.0
            for cname, fmat in class_features.items():
                sims = np.dot(fmat, mf)
                sim = float(np.max(sims)) if len(sims) else -1.0
                if sim >= threshold and sim > best_sim:
                    best_sim, best_class = sim, cname
            if best_class is not None:
                combined[best_class] |= md["mask"]

        # Save one mask file per class (only if non-empty)
        stem = image_path.stem
        for cname, mask in combined.items():
            if mask.any():
                safe_cname = "".join(c if c.isalnum() or c in "-_" else "_" for c in cname)
                _save_mask_rgba(mask, masks_dir, f"{stem}_{safe_cname}")

    def _classes_summary(self, classes: Dict) -> Dict:
        result = {}
        for cname, cdata in classes.items():
            result[cname] = {
                "color": cdata["color"],
                "num_examples": len(cdata["examples"]),
                "examples": [
                    {
                        "id": ex["id"],
                        "x": ex["x"],
                        "y": ex["y"],
                        "area": ex["area"],
                        "sam_score": ex["sam_score"],
                        "mask_index": ex["mask_index"],
                        "contour": ex["contour"],
                    }
                    for ex in cdata["examples"]
                ],
                "predictions": cdata.get("predictions", []),
            }
        return result

    def _persist_state(self, project_fewshot_dir: Path, image_id: str) -> None:
        """Persist mask arrays, features and class metadata to disk."""
        data = self._store.get(image_id)
        if data is None:
            return
        save_dir = Path(project_fewshot_dir) / image_id
        save_dir.mkdir(parents=True, exist_ok=True)

        all_masks = data["all_masks"]
        # masks.npz
        np.savez_compressed(
            save_dir / "masks.npz",
            **{str(i): m["mask"] for i, m in enumerate(all_masks)},
        )
        # features.npy
        feats = np.array([m["feature"] for m in all_masks])
        np.save(save_dir / "features.npy", feats)
        # meta.json  (contours, areas, bboxes)
        meta = [{"contour": m["contour"], "area": m["area"], "bbox": m["bbox"]} for m in all_masks]
        (save_dir / "meta.json").write_text(json.dumps(meta))
        # classes.json  (without numpy arrays)
        classes_serial = {}
        for cname, cdata in data["classes"].items():
            classes_serial[cname] = {
                "color": cdata["color"],
                "examples": [
                    {
                        "id": ex["id"],
                        "x": ex["x"],
                        "y": ex["y"],
                        "mask_index": ex["mask_index"],
                        "feature": ex["feature"].tolist(),
                        "contour": ex["contour"],
                        "area": ex["area"],
                        "sam_score": ex["sam_score"],
                    }
                    for ex in cdata["examples"]
                ],
            }
        (save_dir / "classes.json").write_text(json.dumps(classes_serial))

    def _load_classes(self, project_fewshot_dir: Path, image_id: str) -> Dict:
        """Restore class definitions (with features) from disk if available."""
        classes_path = Path(project_fewshot_dir) / image_id / "classes.json"
        if not classes_path.exists():
            return {}
        try:
            raw = json.loads(classes_path.read_text())
            classes = {}
            for cname, cdata in raw.items():
                examples = []
                for ex in cdata["examples"]:
                    ex = dict(ex)
                    ex["feature"] = np.array(ex["feature"], dtype=np.float32)
                    ex["mask"] = np.zeros((1, 1), dtype=bool)  # placeholder
                    examples.append(ex)
                classes[cname] = {
                    "color": cdata["color"],
                    "examples": examples,
                    "predictions": [],
                    "rejected": set(),
                }
            return classes
        except Exception as e:
            logger.warning(f"Could not restore classes for {image_id}: {e}")
            return {}

    def _load_all_classes(self, project_fewshot_dir: Path) -> Dict:
        """
        Aggregate class definitions (with features) from ALL image directories
        in the fewshot folder.  This way 'process_image' uses examples labeled
        on ANY image, not just the one being analyzed.
        """
        merged: Dict = {}
        fewshot_path = Path(project_fewshot_dir)
        if not fewshot_path.exists():
            return {}
        ex_counter: Dict[str, int] = {}
        for image_dir in sorted(fewshot_path.iterdir()):
            if not image_dir.is_dir():
                continue
            classes_path = image_dir / "classes.json"
            if not classes_path.exists():
                continue
            try:
                raw = json.loads(classes_path.read_text())
                for cname, cdata in raw.items():
                    if cname not in merged:
                        merged[cname] = {
                            "color": cdata["color"],
                            "examples": [],
                            "predictions": [],
                            "rejected": set(),
                        }
                        ex_counter[cname] = 0
                    for ex in cdata["examples"]:
                        ex_copy = dict(ex)
                        ex_copy["feature"] = np.array(ex_copy["feature"], dtype=np.float32)
                        ex_copy["mask"]    = np.zeros((1, 1), dtype=bool)
                        ex_copy["id"]      = ex_counter[cname]
                        ex_counter[cname] += 1
                        merged[cname]["examples"].append(ex_copy)
            except Exception as e:
                logger.warning(f"Could not load classes from {image_dir}: {e}")
        total = sum(len(v["examples"]) for v in merged.values())
        logger.info(f"Loaded {total} examples across {len(merged)} classes from disk.")
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Mask I/O helper (shared with MaskExtractor pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def _save_mask_rgba(mask: np.ndarray, output_dir: Path, stem: str) -> None:
    """Save boolean mask as RGBA PNG compatible with MaskExtractor."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_repeated = np.repeat(np.expand_dims(mask.astype(np.uint8) * 128, 2), 4, axis=2)
    Image.fromarray(mask_repeated.astype(np.uint8), mode="RGBA").save(
        output_dir / f"{stem}_mask_layer.png"
    )


class MaskExtractor:
    """Handles mask extraction using black mask annotations"""
    
    def __init__(self, config: MaskExtractionConfig):
        self.config = config

    def _setup_directories(self, folder: str) -> tuple[Path, str, Path, Path]:
        """Setup and return required directories and image format"""
        img_folder = self.config.pdfimg_output_dir / folder
        img_format = os.listdir(img_folder)[0].split(".")[1]
        mask_folder = self.config.pred_output_dir / f"{folder}_mask"
        output_folder = self.config.pred_output_dir / f"{folder}_card"
        os.makedirs(output_folder, exist_ok=True)
        return img_folder, img_format, mask_folder, output_folder

    #def _process_mask(self, mask_array: np.ndarray) -> np.ndarray:
    #    """Process mask array to get labeled regions"""
    #    thresh = threshold_otsu(mask_array)
    #    bw = closing(mask_array > thresh, square(3))
    #    cleared = clear_border(bw)

    #    return label(cleared)


    def _process_mask(self, image_array: np.ndarray) -> np.ndarray:
        """
        Process PNG with alpha channel to get labeled regions from non-transparent areas.
        """
        # Extract alpha channel (assuming it's the 4th channel)
        if image_array.ndim == 3 and image_array.shape[2] >= 4:
            alpha_channel = image_array[:, :, 3]
        else:
            raise ValueError("Input image doesn't appear to have an alpha channel")
        
        # Create binary mask from alpha channel (non-zero alpha values)
        mask_array = np.where(alpha_channel > 0, 1, 0).astype(np.uint8)
        
        # Only apply Otsu thresholding if we have both foreground and background pixels
        if mask_array.min() < mask_array.max():
            thresh = threshold_otsu(alpha_channel)
            bw = closing(alpha_channel > thresh, square(3))
        else:
            bw = mask_array
        
        # Clear artifacts connected to image border
        cleared = clear_border(bw)
        
        # Label connected regions
        return label(cleared)

    
    def _create_region_mask(self, shape: tuple, region_bbox: tuple, region_mask: np.ndarray) -> np.ndarray:
        """Create full-size mask for a region"""
        minr, minc, maxr, maxc = region_bbox
        mask = np.zeros(shape)
        mask[minr:maxr, minc:maxc] = region_mask
        return mask

    def _expand_mask(self, mask: np.ndarray) -> np.ndarray:
        """Expand mask to match image dimensions"""
        mask_exp = np.expand_dims(mask * 255, axis=-1)
        return np.repeat(mask_exp, 3, axis=-1).astype(np.uint8)


    def _extract_region(self, 
                    region: 'RegionProperties', 
                    mask_array: np.ndarray, 
                    orig_array: np.ndarray, 
                    total_area: int) -> tuple[np.ndarray, tuple] | None:
        """
        Extract region using precise segmentation mask with PIL for image handling
        """
        if region.area < total_area * self.config.min_area_ratio:
            return None

        # Get bounding box coordinates
        minr, minc, maxr, maxc = region.bbox
        
        # Ensure dimensions match before processing
        mask_shape = mask_array.shape[:2]
        orig_shape = orig_array.shape[:2]
        
        if mask_shape != orig_shape:
            # Convert mask array to PIL Image for resizing
            mask_img = Image.fromarray(mask_array)
            # Resize mask to match original image dimensions
            mask_resized = mask_img.resize((orig_array.shape[1], orig_array.shape[0]), 
                                        resample=Image.Resampling.NEAREST)
            # Convert back to numpy array
            mask_array = np.array(mask_resized)
            
            # Recalculate bbox coordinates
            scale_y = orig_shape[0] / mask_shape[0]
            scale_x = orig_shape[1] / mask_shape[1]
            minr = int(minr * scale_y)
            maxr = int(maxr * scale_y)
            minc = int(minc * scale_x)
            maxc = int(maxc * scale_x)
            
            # Convert region mask to PIL Image and resize
            region_mask_img = Image.fromarray(region.image.astype(np.uint8) * 255)
            region_mask_resized = region_mask_img.resize((maxc - minc, maxr - minr), 
                                                        resample=Image.Resampling.NEAREST)
            region_mask = np.array(region_mask_resized) > 0
        else:
            region_mask = region.image

        # Create full-size mask
        full_mask = np.zeros_like(orig_array[:,:,0], dtype=bool)
        full_mask[minr:maxr, minc:maxc] = region_mask
        
        # Expand mask to match image dimensions
        mask_exp = np.expand_dims(full_mask, axis=-1).astype(np.uint8)
        mask_exp = np.repeat(mask_exp, 3, axis=-1)
        
        # Apply mask to original image
        masked_img = np.where(mask_exp == 0, 255, orig_array)
        
        # Crop to bounding box
        cropped = masked_img[minr:maxr, minc:maxc]
        
        return cropped, (minc, minr, maxc, maxr)

    def _save_metadata(self, 
                      metadata: list[tuple], 
                      annotations: list[tuple], 
                      output_folder: Path) -> None:
        """Save extraction metadata to CSV files"""
        if not metadata:
            return
            
        pd.DataFrame(metadata, columns=["file", "mask_file"]).to_csv(
            output_folder / "mask_info.csv", index=False
        )
        pd.DataFrame(annotations, columns=["bbox", "mask_file"]).to_csv(
            output_folder / "mask_info_annots.csv", index=False
        )

    def extract_masks(self, drop_folder_review: str) -> str:
        """Extract masks from images in folder"""
        try:
            # Setup directories
            img_folder, img_format, mask_folder, output_folder = self._setup_directories(drop_folder_review)
            
            metadata = []
            annotations = []

            # Get list of files to process
            mask_files = os.listdir(mask_folder)
            total_files = len(mask_files)
            
            # Process each mask file
            for idx, file in enumerate(mask_files, 1):
                print(f"Processing mask {idx}/{total_files}: {file}")
                
                base_filename = file.split(".")[0].replace("_mask_layer", "")
                
                # Load images
                mask_array = np.array(Image.open(mask_folder / file))
                orig_array = np.array(Image.open(img_folder / f"{base_filename}.{img_format}"))
                
                # Process mask and get labeled regions
                label_image = self._process_mask(mask_array)
                total_area = mask_array.size
                
                # Process each region
                for i, region in enumerate(regionprops(label_image)):
                    result = self._extract_region(region, mask_array, orig_array, total_area)
                    if result is None:
                        continue
                        
                    cropped, bbox = result
                    output_filename = f"{base_filename}_mask_layer_{i}.png"
                    
                    # Save cropped image
                    ### add some white space around the cropped image
                    cropped = np.pad(cropped, ((50, 50), (50, 50), (0, 0)), mode='constant', constant_values=255)
                    Image.fromarray(cropped).save(output_folder / output_filename)
                    
                    # Store metadata
                    metadata.append((base_filename, f"{base_filename}_mask_layer_{i}"))
                    annotations.append((bbox, output_filename))

            # Save metadata
            self._save_metadata(metadata, annotations, output_folder)
            
            if metadata:
                return f"Successfully extracted {len(metadata)} masks from '{drop_folder_review}'"
            return "No masks were extracted. Check if masks are properly drawn."

        except Exception as e:
            print(f"Error in mask extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error extracting masks: {str(e)}"

    def extract_masks_from_project(self, masks_path: str, cards_path: str) -> str:
        """Extract cards from masks in a project"""
        try:
            masks_path = Path(masks_path)
            cards_path = Path(cards_path)
            
            if not masks_path.exists():
                return "Masks folder not found"
            
            # Create cards folder
            os.makedirs(cards_path, exist_ok=True)
            
            # Get all mask files
            mask_files = [f.name for f in masks_path.iterdir() 
                         if f.name.endswith('_mask_layer.png')]
            
            if not mask_files:
                return "No mask files found. Apply a model first."
            
            metadata = []
            annotations = []
            
            total_files = len(mask_files)
            
            # Process each mask file
            for idx, file in enumerate(mask_files, 1):
                print(f"Processing mask {idx}/{total_files}: {file}")
                
                base_filename = file.replace("_mask_layer.png", "")
                
                # Load mask
                mask_array = np.array(Image.open(masks_path / file))
                
                # Try to find corresponding original image
                # Look for image in parent's images folder
                orig_image_path = masks_path.parent / 'images' / f"{base_filename}.jpg"
                if not orig_image_path.exists():
                    orig_image_path = masks_path.parent / 'images' / f"{base_filename}.png"
                
                if not orig_image_path.exists():
                    print(f"Warning: Original image not found for {base_filename}")
                    continue
                
                orig_array = np.array(Image.open(orig_image_path))
                
                # Process mask and get labeled regions
                label_image = self._process_mask(mask_array)
                total_area = mask_array.size
                
                # Process each region
                for i, region in enumerate(regionprops(label_image)):
                    result = self._extract_region(region, mask_array, orig_array, total_area)
                    if result is None:
                        continue
                        
                    cropped, bbox = result
                    output_filename = f"{base_filename}_mask_layer_{i}.png"
                    
                    # Save cropped image with padding
                    cropped = np.pad(cropped, ((50, 50), (50, 50), (0, 0)), mode='constant', constant_values=255)
                    Image.fromarray(cropped).save(cards_path / output_filename)
                    
                    # Store metadata
                    metadata.append((base_filename, f"{base_filename}_mask_layer_{i}"))
                    annotations.append((bbox, output_filename))

            # Save metadata
            self._save_metadata(metadata, annotations, cards_path)
            
            if metadata:
                return f"Successfully extracted {len(metadata)} cards from project masks"
            return "No cards were extracted. Check if masks are properly drawn."

        except Exception as e:
            print(f"Error in mask extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error extracting masks: {str(e)}"
    
class TabularProcessor:
    """Handles tabular data viewing and editing for extracted masks"""
    
    def __init__(self, config: TabularConfig):
        self.pdfimg_output_dir = Path(config.pdfimg_output_dir).resolve()
        self.pred_output_dir = Path(config.pred_output_dir).resolve()
        self._current_file = None

    def get_results_folders(self) -> List[str]:
        """Get list of folders containing results with validation"""
        try:
            folder_list = [f for f in os.listdir(self.pred_output_dir) 
                          if f.endswith('_card') and not f.endswith('transformed_card')]
            
            # Validate each folder contains required files
            valid_folders = []
            for folder in folder_list:
                folder_path = self.pred_output_dir / folder
                if folder_path.is_dir():
                    mask_info = folder_path / "mask_info.csv"
                    mask_info_annots = folder_path / "mask_info_annots.csv"
                    if mask_info.exists() and mask_info_annots.exists():
                        valid_folders.append(folder)
                        print(f"Valid folder found: {folder}")
            
            return valid_folders
            
        except Exception as e:
            print(f"Error in get_results_folders: {str(e)}")
            return []

    def convert_bbox(self, bbox_str: str) -> tuple:
        """Convert bbox string to tuple with error handling"""
        try:
            # Remove parentheses and split
            bbox = bbox_str.strip('()').split(',')
            return tuple(int(float(coord)) for coord in bbox)
        except Exception as e:
            print(f"Error converting bbox {bbox_str}: {str(e)}")
            return (0, 0, 0, 0)

    def create_annotation_tuple(self, df: pd.DataFrame, image_name: str) -> List[tuple]:
        """Create list of annotation tuples with validation"""
        try:
            # Filter dataframe for current image
            df_selected = df[df['image_name'] == image_name].copy()
            
            if df_selected.empty:
                print(f"No annotations found for image {image_name}")
                return []
            
            # Create annotation tuples with error handling
            annotations = []
            for _, row in df_selected.iterrows():
                try:
                    bbox = self.convert_bbox(row['bbox'])
                    mask_id = str(row['ID']).strip()  # Ensure ID is string and clean
                    if all(isinstance(x, (int, float)) for x in bbox):
                        annotations.append((bbox, mask_id))
                except Exception as e:
                    print(f"Error processing annotation row: {str(e)}")
                    continue
                    
            return annotations
            
        except Exception as e:
            print(f"Error creating annotations: {str(e)}")
            return []

    def image_selection(self, folder: str, img_num: int) -> tuple:
        """Select and prepare image and associated data for display"""
        try:
            if not folder:
                return None, img_num, pd.DataFrame()

            # Setup paths
            context = folder.split("_card")[0]
            folder_mask_path = (self.pred_output_dir / f"{context}_mask").resolve()
            folder_img_path = (self.pdfimg_output_dir / context).resolve()
            csv_path = (self.pred_output_dir / folder).resolve()

            print(f"\nProcessing paths:")
            print(f"Mask folder: {folder_mask_path}")
            print(f"Image folder: {folder_img_path}")
            print(f"CSV folder: {csv_path}")

            # Validate paths existence
            if not all(p.exists() for p in [folder_mask_path, folder_img_path, csv_path]):
                print("Missing required folders")
                return None, img_num, pd.DataFrame()

            # Load CSV files
            try:
                mask_info_path = csv_path / "mask_info.csv"
                mask_info_annots_path = csv_path / "mask_info_annots.csv"

                print(f"\nReading CSV files:")
                print(f"mask_info.csv: {mask_info_path}")
                print(f"mask_info_annots.csv: {mask_info_annots_path}")

                if not mask_info_path.exists() or not mask_info_annots_path.exists():
                    print("Required CSV files not found")
                    return None, img_num, pd.DataFrame()

                # Read CSVs with explicit dtypes
                df = pd.read_csv(mask_info_path).fillna('')
                df_annots = pd.read_csv(mask_info_annots_path)
                
                # Clean and prepare annotation data
                df_annots['image_name'] = df_annots['mask_file'].apply(
                    lambda x: x.split('_mask')[0] if isinstance(x, str) else '')
                df_annots['ID'] = df_annots['mask_file'].apply(
                    lambda x: x.split('layer_')[1] if isinstance(x, str) else '')

            except Exception as e:
                print(f"Error reading CSV files: {str(e)}")
                return None, img_num, pd.DataFrame()

            # Get valid images with corresponding masks
            images = []
            for f in os.listdir(folder_img_path):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                mask_file = f.split(".")[0] + "_mask_layer.png"
                if os.path.exists(folder_mask_path / mask_file):
                    images.append(f)

            if not images:
                print("No valid images found")
                return None, img_num, pd.DataFrame()

            # Get current image
            img_num = max(0, min(img_num, len(images) - 1))
            current_image = images[img_num]
            image_base_name = current_image.split(".")[0]
            
            print(f"\nProcessing image: {image_base_name}")
            
            # Store current file
            self._current_file = image_base_name

            # Prepare display data
            df_subset = df[df["file"] == image_base_name].copy()
            
            if df_subset.empty:
                print(f"No data found for image {image_base_name}")
                return None, img_num, pd.DataFrame()

            df_subset["ID"] = df_subset["mask_file"].apply(lambda x: x.split('layer_')[1] if isinstance(x, str) else '')
            
            # Clean up columns
            drop_cols = [col for col in ["mask_file", "file"] if col in df_subset.columns]
            if drop_cols:
                df_subset.drop(columns=drop_cols, inplace=True)
            
            # Reorder columns
            columns_order = ["ID"] + [col for col in df_subset.columns if col != "ID"]
            df_display = df_subset[columns_order]

            # Create image with annotations
            try:
                img_path = folder_img_path / current_image
                if not img_path.exists():
                    print(f"Image file not found: {img_path}")
                    return None, img_num, df_display

                with Image.open(img_path) as img:
                    original_size = img.size  # Save original dimensions
                    img.thumbnail((1200, 1200))
                    scale_x = img.size[0] / original_size[0]
                    scale_y = img.size[1] / original_size[1]
                    image = np.asarray(img, dtype=np.uint8)
                    
                # Get original annotations and scale them
                original_annotations = self.create_annotation_tuple(df_annots, image_base_name)
                scaled_annotations = []
                    
                for bbox, mask_id in original_annotations:
                    scaled_bbox = (
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y)
                    )
                    scaled_annotations.append((scaled_bbox, mask_id))
                
                return (
                    gr.AnnotatedImage(value=[image, scaled_annotations]), 
                    img_num, 
                    df_display
                )
                
            except Exception as e:
                print(f"Error creating annotated image: {str(e)}")
                return None, img_num, df_display

        except Exception as e:
            print(f"Error in image selection: {str(e)}")
            return None, img_num, pd.DataFrame()

    def save_table(self, table: pd.DataFrame, folder: str) -> None:
        """Save table with robust error handling"""
        try:
            if self._current_file is None:
                print("No file currently selected")
                return

            csv_path = self.pred_output_dir / folder / "mask_info.csv"
            if not csv_path.exists():
                print(f"CSV file not found: {csv_path}")
                return

            # Read existing table
            existing_table = pd.read_csv(csv_path)
            
            # Add back file information
            table = table.copy()
            table['file'] = self._current_file
            table["mask_file"] = table["file"] + "_mask_layer_" + table["ID"]
            
            if "ID" in table.columns:
                table.drop(columns=["ID"], inplace=True)
            
            # Update existing table
            existing_table.set_index("mask_file", inplace=True)
            table.set_index("mask_file", inplace=True)

            # Update values
            for col in table.columns:
                if col not in existing_table.columns:
                    existing_table[col] = pd.NA
                existing_table.loc[table.index, col] = table[col]

            # Clean up and save
            existing_table.reset_index(inplace=True)
            existing_table = existing_table.loc[:, ~existing_table.columns.str.contains('Header')]
            existing_table.to_csv(csv_path, index=False)
            
            print(f"Table saved successfully to {csv_path}")
            
        except Exception as e:
            print(f"Error saving table: {str(e)}")
            


@dataclass
class ExportConfig:
    """Configuration for final export processing"""
    pred_output_dir: Path
    export_pdf: bool = False
    pdf_page_size: str = 'A4'
    scale_factor: float = 1.0

class ExportProcessor:
    """Handles final export processing with PDF export capability"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.pdf_exporter = None  # Will be initialized when needed

    def export_results(self, folder: str, acronym: str, export_pdf: bool = False, 
                        page_size: str = 'A4', scale_factor: float = 1.0) -> str:
            """
            Export processed images and metadata with optional PDF export
            """
            try:
                # Setup paths
                base_folder = folder.split("_card")[0]
                source_folder = self.config.pred_output_dir / f"{base_folder}_transformed_card"
                export_folder = self.config.pred_output_dir / f"{acronym}"
                
                if not source_folder.exists():
                    return f"Transformed folder not found. Please process images first."

                merged_annotations_path = source_folder / "merged_annotations.csv"
                    
                if not merged_annotations_path.exists():
                    return "Merged annotations file not found. Please merge annotations first."

                try:
                    # Load merged_annotations
                    metadata = pd.read_csv(merged_annotations_path)
                    
                    # Create the export folder
                    os.makedirs(export_folder, exist_ok=True)

                    # Create new sequential IDs and track exported image paths
                    image_data = []  # List to store (path, new_id) tuples
                    metadata['new_id'] = [f"{acronym}_{i+1}" for i in range(len(metadata))]
                    metadata.set_index('new_id', inplace=True)
                    
                    # Copy transformed images with new names
                    copied_count = 0
                    for idx, row in metadata.iterrows():
                        try:
                            source_image = source_folder / f"{row['filename']}.png"
                            if source_image.exists():
                                dest_image = export_folder / f"{idx}.png"
                                shutil.copy2(source_image, dest_image)
                                image_data.append((str(dest_image), idx))  # Store path and new_id
                                copied_count += 1
                        except Exception as e:
                            print(f"Error copying image {idx}: {str(e)}")
                            continue

                    # Clean up and save metadata
                    metadata = metadata.drop('filename', axis=1, errors='ignore')
                    metadata.to_csv(export_folder / f"{acronym}_metadata.csv")
                    
                    # Generate PDF if requested
                    pdf_message = ""
                    if export_pdf and image_data:
                        pdf_path = export_folder / f"{acronym}_catalog.pdf"
                        pdf_exporter = PDFExporter(
                            page_size=page_size,
                            scale_factor=scale_factor
                        )
                        if pdf_exporter.generate_pdf(str(pdf_path), image_data):
                            pdf_message = f" PDF catalog generated at {pdf_path}."
                        else:
                            pdf_message = " Warning: PDF generation failed."

                    if copied_count == 0:
                        return "Warning: No images were exported."
                        
                    return (f"Export complete: {copied_count} images exported to {export_folder} "
                        f"with prefix '{acronym}_'.{pdf_message}")
                    
                except Exception as e:
                    print(f"Detailed error: {str(e)}")
                    return f"Error processing data: {str(e)}"
                
            except Exception as e:
                print(f"Export error: {str(e)}")
                return f"Error during export: {str(e)}"
        


from typing import List, Dict, Tuple
from PIL import Image
from reportlab.lib import pagesizes
from reportlab.pdfgen import canvas
import numpy as np

class LayoutNode:
    """Tree node representing available space in the layout"""
    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.used = False
        self.down = None
        self.right = None
        self.image_info = None

    def fits(self, width: float, height: float) -> bool:
        """Check if an image fits in this node"""
        return (not self.used and 
                width <= self.width and 
                height <= self.height)

class PDFLayoutOptimizer:
    """Optimizes PDF layout using binary tree bin packing algorithm"""
    
    def __init__(self, page_width: float, page_height: float, margin: float = 50):
        self.page_width = page_width - (2 * margin)
        self.page_height = page_height - (2 * margin)
        self.margin = margin
        self.min_padding = 10  # Minimum padding between images
        self.label_height = 20  # Height reserved for labels

    def find_node(self, root: LayoutNode, width: float, height: float) -> LayoutNode:
        """Find a node that can accommodate the given dimensions"""
        if not root:
            return None
            
        if root.used:
            # Try finding space in existing splits
            node = self.find_node(root.right, width, height)
            if not node:
                node = self.find_node(root.down, width, height)
            return node
            
        elif root.fits(width, height):
            return root
            
        return None

    def split_node(self, node: LayoutNode, width: float, height: float) -> None:
        """Split a node to accommodate an image and create remaining space"""
        node.used = True
        
        # Create space below current image
        node.down = LayoutNode(
            node.x,
            node.y + height + self.min_padding,
            node.width,
            node.height - height - self.min_padding
        )
        
        # Create space to the right of current image
        node.right = LayoutNode(
            node.x + width + self.min_padding,
            node.y,
            node.width - width - self.min_padding,
            height
        )

    def optimize_page_layout(self, images: List[Dict]) -> List[List[Dict]]:
        """Optimize layout across multiple pages using bin packing"""
        pages = []
        current_images = images.copy()
        
        while current_images:
            # Initialize new page
            root = LayoutNode(self.margin, self.margin, self.page_width, self.page_height)
            page_layout = []
            remaining_images = []
            
            for img in current_images:
                # Account for label height in total height
                total_height = img['height'] + self.label_height + self.min_padding
                
                # Find space for image
                node = self.find_node(root, img['width'], total_height)
                
                if node:
                    # Place image and split remaining space
                    self.split_node(node, img['width'], total_height)
                    node.image_info = {
                        'path': img['path'],
                        'new_id': img['new_id'],
                        'x': node.x,
                        'y': node.y,
                        'width': img['width'],
                        'height': img['height'],
                        'label_y': node.y + img['height'] + 5
                    }
                    page_layout.append(node.image_info)
                else:
                    remaining_images.append(img)
            
            pages.append(page_layout)
            current_images = remaining_images
            
        return pages

    def pack_images(self, image_paths: List[tuple], scale_factor: float = 1.0) -> List[List[Dict]]:
        """Process images and optimize their layout"""
        # Get image dimensions and create scaled info
        ######
        px_to_pt = 72.0 / 300
        ###### 
        image_info = []
        
        for img_path, new_id in image_paths:
            with Image.open(img_path) as img:
                w, h = img.size


                ############
                # Convert pixel dimensions to PDF points
                w = w * px_to_pt
                h = h * px_to_pt
                ###########


                # Scale dimensions
                scaled_w = w * scale_factor
                scaled_h = h * scale_factor

                
                # Ensure scaled image fits on page
                if scaled_w > self.page_width:
                    scale = self.page_width / scaled_w
                    scaled_w *= scale
                    scaled_h *= scale
                
                if scaled_h > (self.page_height - self.label_height):
                    scale = (self.page_height - self.label_height) / scaled_h
                    scaled_w *= scale
                    scaled_h *= scale
                
                image_info.append({
                    'path': img_path,
                    'new_id': new_id,
                    'width': scaled_w,
                    'height': scaled_h,
                    'area': scaled_w * scaled_h
                })
        
        # Sort images by area for better packing
        image_info.sort(key=lambda x: x['area'], reverse=True)
        
        # Optimize layout
        return self.optimize_page_layout(image_info)

class PDFExporter:
    """Handles PDF generation with optimized layout"""
    
    PAGE_SIZES = {
        'A4': pagesizes.A4,
        'A3': pagesizes.A3,
        'A5': pagesizes.A5,
        'LETTER': pagesizes.LETTER,
        'LEGAL': pagesizes.LEGAL
    }
    
    def __init__(self, page_size: str = 'A4', margin: int = 50, scale_factor: float = 1.0):

        self.page_size = self.PAGE_SIZES.get(page_size, pagesizes.A4)
        self.margin = margin
        self.scale_factor = scale_factor
        self.optimizer = PDFLayoutOptimizer(
            page_width=self.page_size[0],
            page_height=self.page_size[1],
            margin=margin
        )
        
      

    def generate_pdf(self, output_path: str, image_data: List[tuple]) -> bool:
        """Generate PDF with optimized image layout"""
        try:
            # Create PDF canvas
            c = canvas.Canvas(output_path, pagesize=self.page_size)
            
            # Set up font for labels
            c.setFont("Helvetica", 10)
            
            # Get optimized layout
            pages = self.optimizer.pack_images(image_data, self.scale_factor)
            
            # Generate each page
            for page in pages:
                for img_info in page:
                    # Draw image
                    c.drawImage(
                        img_info['path'],
                        img_info['x'],
                        img_info['y'],
                        width=img_info['width'],
                        height=img_info['height'],
                        preserveAspectRatio=True
                    )
                    
                    # Draw centered label
                    text = img_info['new_id']
                    text_width = c.stringWidth(text, "Helvetica", 10)
                    center_x = img_info['x'] + (img_info['width'] / 2) - (text_width / 2)
                    
                    c.drawString(
                        center_x,
                        img_info['label_y'],
                        text
                    )
                
                c.showPage()
            
            c.save()
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return False