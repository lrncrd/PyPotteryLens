"""
Crop Processor for PyPotteryLens

This module provides automatic and manual cropping capabilities for pottery images.
The auto-crop feature can detect and remove section/profile drawings from pottery images.
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class CropProcessor:
    """Handles automatic and manual cropping of pottery images."""

    def __init__(self):
        """Initialize the crop processor."""
        self._has_skimage = False
        self._has_cv2 = False
        try:
            from skimage import feature, filters
            self._has_skimage = True
        except ImportError:
            logger.warning("scikit-image not available, auto-crop will use fallback method")

        try:
            import cv2
            self._has_cv2 = True
        except ImportError:
            logger.warning("OpenCV not available, some features will be limited")

    def auto_remove_section_preview(self, image: np.ndarray) -> dict:
        """
        Analyze image and return preview of both halves without cropping.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary with:
            - split_x: detected split position
            - left_preview: base64 of left half
            - right_preview: base64 of right half
            - complexity_left: complexity score of left half
            - complexity_right: complexity score of right half
            - recommended_side: 'left' or 'right' based on analysis
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        height, width = image.shape[:2]

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        # Find the vertical split line using multiple methods
        split_x = self._find_vertical_split_advanced(gray, width // 2)

        # Ensure minimum width for both halves
        min_width = width // 4
        if split_x < min_width:
            split_x = width // 2
        elif split_x > width - min_width:
            split_x = width // 2

        # Split image
        left_half = image[:, :split_x]
        right_half = image[:, split_x:]

        # Analyze complexity of both halves
        left_complexity = self._compute_complexity_advanced(left_half)
        right_complexity = self._compute_complexity_advanced(right_half)

        # Convert previews to base64
        import base64
        from io import BytesIO

        def array_to_base64(arr):
            img = Image.fromarray(arr)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        recommended = 'left' if left_complexity <= right_complexity else 'right'
        confidence = float(abs(left_complexity - right_complexity) / max(left_complexity, right_complexity, 0.001))

        return {
            'split_x': int(split_x),
            'width': int(width),
            'height': int(height),
            'left_preview': f'data:image/png;base64,{array_to_base64(left_half)}',
            'right_preview': f'data:image/png;base64,{array_to_base64(right_half)}',
            'complexity_left': float(left_complexity),
            'complexity_right': float(right_complexity),
            'recommended_side': recommended,
            'confidence': confidence
        }

    def auto_remove_section(self, image: np.ndarray, keep_side: str = 'auto') -> Tuple[np.ndarray, dict]:
        """
        Automatically detect and remove section/profile drawing from pottery image.

        Pottery technical drawings often have the vessel profile/section on one half
        and the silhouette on the other. This method detects the division and keeps
        the silhouette half.

        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
            keep_side: 'auto' (detect), 'left', or 'right'

        Returns:
            Tuple of (cropped_image, metadata_dict)
            metadata_dict contains: split_x, kept_side, complexity_left, complexity_right
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        height, width = image.shape[:2]

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        # Find the vertical split line using advanced method
        split_x = self._find_vertical_split_advanced(gray, width // 2)

        # Ensure minimum width for both halves
        min_width = width // 4
        if split_x < min_width:
            split_x = width // 2
        elif split_x > width - min_width:
            split_x = width // 2

        # Split image
        left_half = image[:, :split_x]
        right_half = image[:, split_x:]

        # Analyze complexity of both halves
        left_complexity = self._compute_complexity_advanced(left_half)
        right_complexity = self._compute_complexity_advanced(right_half)

        metadata = {
            'split_x': int(split_x),
            'complexity_left': float(left_complexity),
            'complexity_right': float(right_complexity),
            'width': int(width),
            'height': int(height)
        }

        # Determine which side to keep
        if keep_side == 'auto':
            # Section drawings typically have more internal detail/lines
            # Keep the less complex half (pottery silhouette)
            if left_complexity <= right_complexity:
                metadata['kept_side'] = 'left'
                return left_half, metadata
            else:
                metadata['kept_side'] = 'right'
                return right_half, metadata
        elif keep_side == 'left':
            metadata['kept_side'] = 'left'
            return left_half, metadata
        else:
            metadata['kept_side'] = 'right'
            return right_half, metadata

    def _find_vertical_split_advanced(self, gray: np.ndarray, mid_x: int) -> int:
        """
        Find the vertical split line using multiple detection methods.

        Methods used:
        1. Edge detection (Canny) - looks for strong vertical edges
        2. Dark line detection - pottery drawings often have a solid black split line
        3. Symmetry analysis - the split should be at the axis of symmetry
        4. Gradient analysis - looks for sudden changes in content
        """
        height, width = gray.shape

        # Define search region around center (wider for better detection)
        search_width = min(150, width // 3)
        left_bound = max(0, mid_x - search_width)
        right_bound = min(width, mid_x + search_width)

        candidates = []
        weights = []

        # Method 1: Edge detection
        if self._has_skimage:
            try:
                from skimage import feature
                edges = feature.canny(gray, sigma=1.5)
                center_region = edges[:, left_bound:right_bound]

                # Sum edges vertically to find strongest vertical line
                vertical_sum = np.sum(center_region, axis=0)

                if vertical_sum.size > 0:
                    # Look for peaks (strong vertical lines)
                    threshold = height * 0.15  # At least 15% of height should be edge
                    peaks = np.where(vertical_sum > threshold)[0]

                    if len(peaks) > 0:
                        # Find the peak closest to center
                        center_of_region = search_width
                        closest_peak = peaks[np.argmin(np.abs(peaks - center_of_region))]
                        candidates.append(left_bound + closest_peak)
                        weights.append(vertical_sum[closest_peak] / height)
            except Exception as e:
                logger.warning(f"Edge detection failed: {e}")

        # Method 2: Dark line detection (solid black vertical line)
        center_region = gray[:, left_bound:right_bound]
        col_means = np.mean(center_region, axis=0)
        col_stds = np.std(center_region, axis=0)

        # Look for dark columns with low variance (solid lines)
        dark_threshold = np.percentile(col_means, 10)  # Bottom 10% darkness
        dark_cols = np.where(col_means < dark_threshold)[0]

        if len(dark_cols) > 0:
            # Among dark columns, find the one with lowest std (most uniform = solid line)
            best_dark = dark_cols[np.argmin(col_stds[dark_cols])]
            candidates.append(left_bound + best_dark)
            weights.append(1.0 - col_means[best_dark] / 255.0)  # Darker = higher weight

        # Method 3: Symmetry analysis
        try:
            symmetry_scores = []
            for x in range(left_bound, right_bound, 5):  # Check every 5 pixels
                # Compare left and right halves around this x
                margin = min(x, width - x, 100)
                if margin < 20:
                    continue
                left_part = gray[:, x-margin:x]
                right_part = gray[:, x:x+margin]
                right_part_flipped = np.flip(right_part, axis=1)

                # Compute similarity (lower = more symmetric)
                diff = np.abs(left_part.astype(float) - right_part_flipped.astype(float))
                symmetry_score = np.mean(diff)
                symmetry_scores.append((x, symmetry_score))

            if symmetry_scores:
                best_symmetry = min(symmetry_scores, key=lambda x: x[1])
                if best_symmetry[1] < 50:  # Good symmetry threshold
                    candidates.append(best_symmetry[0])
                    weights.append(1.0 - best_symmetry[1] / 100.0)
        except Exception as e:
            logger.warning(f"Symmetry analysis failed: {e}")

        # Method 4: Content gradient (sudden change in image content)
        try:
            # Compute horizontal projection (sum of each column)
            projection = np.sum(255 - gray, axis=0)  # Invert so content is positive

            # Smooth and find gradient
            from scipy.ndimage import gaussian_filter1d
            projection_smooth = gaussian_filter1d(projection.astype(float), sigma=10)
            gradient = np.abs(np.gradient(projection_smooth))

            # Find peaks in gradient within search region
            region_gradient = gradient[left_bound:right_bound]
            if len(region_gradient) > 0:
                peak_idx = np.argmax(region_gradient)
                if region_gradient[peak_idx] > np.mean(gradient) * 2:
                    candidates.append(left_bound + peak_idx)
                    weights.append(min(region_gradient[peak_idx] / np.max(gradient), 1.0))
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Gradient analysis failed: {e}")

        # Combine candidates using weighted average
        if candidates:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            final_x = int(np.average(candidates, weights=weights))
            return final_x

        # Fallback to simple center
        return mid_x

    def _compute_complexity_advanced(self, image: np.ndarray) -> float:
        """
        Compute the internal complexity of an image region using multiple metrics.

        Higher complexity indicates more internal detail (like section drawings).
        Lower complexity indicates simpler shapes (like silhouettes).
        """
        if image is None or image.size == 0:
            return 0.0

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        complexity_scores = []

        # Method 1: Edge density (using Canny or gradient)
        if self._has_skimage:
            try:
                from skimage import feature
                edges = feature.canny(gray, sigma=1.0)
                area = gray.shape[0] * gray.shape[1]
                edge_density = np.sum(edges) / area if area > 0 else 0.0
                complexity_scores.append(edge_density * 100)  # Scale to 0-100 range
            except Exception:
                pass

        if not complexity_scores:
            # Fallback: gradient magnitude
            grad_x = np.abs(np.diff(gray.astype(float), axis=1))
            grad_y = np.abs(np.diff(gray.astype(float), axis=0))
            area = gray.shape[0] * gray.shape[1]
            grad_complexity = (np.sum(grad_x) + np.sum(grad_y)) / area if area > 0 else 0.0
            complexity_scores.append(grad_complexity)

        # Method 2: Internal line density
        # Section drawings have internal lines (hatching, cross-sections)
        # Silhouettes are mostly solid black
        try:
            # Threshold to get dark pixels (content)
            content_mask = gray < 200
            content_area = np.sum(content_mask)

            if content_area > 0:
                # Within content, look for white/light pixels (internal lines/spaces)
                internal_light = np.sum((gray > 100) & content_mask)
                internal_ratio = internal_light / content_area
                complexity_scores.append(internal_ratio * 100)
        except Exception:
            pass

        # Method 3: Texture variance
        try:
            # High variance in local regions indicates texture/detail
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(gray.astype(float), size=15)
            local_var = uniform_filter(gray.astype(float)**2, size=15) - local_mean**2
            texture_complexity = np.mean(local_var) / 1000  # Normalize
            complexity_scores.append(min(texture_complexity, 100))
        except ImportError:
            pass
        except Exception:
            pass

        # Return average of all complexity measures
        return np.mean(complexity_scores) if complexity_scores else 0.0

    def _find_vertical_split(self, gray: np.ndarray, mid_x: int) -> int:
        """Legacy method - redirects to advanced version."""
        return self._find_vertical_split_advanced(gray, mid_x)

    def _compute_complexity(self, image: np.ndarray) -> float:
        """Legacy method - redirects to advanced version."""
        return self._compute_complexity_advanced(image)

    def manual_crop(self, image: np.ndarray, rect: Dict[str, int]) -> np.ndarray:
        """
        Manually crop image using rectangle selection.

        Args:
            image: Input image as numpy array
            rect: Dictionary with 'x', 'y', 'width', 'height' keys
                  All values should be in pixels

        Returns:
            Cropped image as numpy array
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        x = int(rect.get('x', 0))
        y = int(rect.get('y', 0))
        w = int(rect.get('width', image.shape[1]))
        h = int(rect.get('height', image.shape[0]))

        # Validate bounds
        height, width = image.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        return image[y:y+h, x:x+w].copy()

    def polygon_crop(self, image: np.ndarray, points: List[Tuple[int, int]],
                     fill_background: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[np.ndarray, dict]:
        """
        Crop image using a freehand polygon selection.

        Args:
            image: Input image as numpy array
            points: List of (x, y) tuples defining the polygon vertices
            fill_background: RGB color to fill outside the polygon

        Returns:
            Tuple of (cropped_image, metadata_dict)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")

        height, width = image.shape[:2]

        # Create mask from polygon
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, fill=255, outline=255)
        mask_array = np.array(mask)

        # Find bounding box of polygon
        points_array = np.array(points)
        x_min, y_min = int(np.min(points_array[:, 0])), int(np.min(points_array[:, 1]))
        x_max, y_max = int(np.max(points_array[:, 0])), int(np.max(points_array[:, 1]))

        # Add padding
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        # Create output image with background color
        if len(image.shape) == 3:
            output = np.full_like(image, fill_background)
        else:
            output = np.full_like(image, fill_background[0])

        # Apply mask
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                output[:, :, c] = np.where(mask_array > 0, image[:, :, c], output[:, :, c])
        else:
            output = np.where(mask_array > 0, image, output)

        # Crop to bounding box
        cropped = output[y_min:y_max, x_min:x_max].copy()

        metadata = {
            'polygon_points': [(int(p[0]), int(p[1])) for p in points],
            'bounding_box': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'original_size': [int(width), int(height)],
            'cropped_size': [int(x_max - x_min), int(y_max - y_min)]
        }

        return cropped, metadata

    def freehand_crop(self, image: np.ndarray, path_points: List[Tuple[int, int]],
                      smoothing: int = 3, fill_background: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[np.ndarray, dict]:
        """
        Crop image using a freehand drawn path (smoothed).

        Args:
            image: Input image as numpy array
            path_points: List of (x, y) tuples from freehand drawing
            smoothing: Amount of smoothing to apply (higher = smoother)
            fill_background: RGB color to fill outside the selection

        Returns:
            Tuple of (cropped_image, metadata_dict)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        if len(path_points) < 3:
            raise ValueError("Path must have at least 3 points")

        # Smooth the path if needed
        if smoothing > 1 and len(path_points) > smoothing * 2:
            try:
                from scipy.ndimage import uniform_filter1d
                points_array = np.array(path_points)
                smoothed_x = uniform_filter1d(points_array[:, 0].astype(float), size=smoothing)
                smoothed_y = uniform_filter1d(points_array[:, 1].astype(float), size=smoothing)
                path_points = [(int(x), int(y)) for x, y in zip(smoothed_x, smoothed_y)]
            except ImportError:
                # Simple moving average fallback
                def smooth_coords(coords, window):
                    smoothed = []
                    for i in range(len(coords)):
                        start = max(0, i - window // 2)
                        end = min(len(coords), i + window // 2 + 1)
                        smoothed.append(int(np.mean(coords[start:end])))
                    return smoothed

                xs = [p[0] for p in path_points]
                ys = [p[1] for p in path_points]
                smoothed_xs = smooth_coords(xs, smoothing)
                smoothed_ys = smooth_coords(ys, smoothing)
                path_points = list(zip(smoothed_xs, smoothed_ys))

        # Use polygon crop with the smoothed path
        return self.polygon_crop(image, path_points, fill_background)

    def crop_to_content(self, image: np.ndarray, padding: int = 10,
                        threshold: int = 250) -> Tuple[np.ndarray, dict]:
        """
        Automatically crop image to its content, removing white/empty borders.

        Args:
            image: Input image as numpy array
            padding: Pixels of padding to keep around content
            threshold: Pixel value above which is considered background (0-255)

        Returns:
            Tuple of (cropped_image, metadata_dict)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Find non-background pixels
        mask = gray < threshold

        if not np.any(mask):
            # All background, return original
            return image, {'cropped': False}

        # Find bounding box of content
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        # Add padding
        height, width = image.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(height, y_max + padding + 1)
        x_min = max(0, x_min - padding)
        x_max = min(width, x_max + padding + 1)

        cropped = image[y_min:y_max, x_min:x_max].copy()

        metadata = {
            'cropped': True,
            'original_size': [int(width), int(height)],
            'crop_box': [int(x_min), int(y_min), int(x_max), int(y_max)]
        }

        return cropped, metadata

    def process_image_file(self, input_path: Union[str, Path],
                           output_path: Union[str, Path],
                           mode: str = 'auto',
                           rect: Optional[Dict[str, int]] = None,
                           points: Optional[List[Tuple[int, int]]] = None,
                           keep_side: str = 'auto') -> dict:
        """
        Process an image file with cropping.

        Args:
            input_path: Path to input image
            output_path: Path to save cropped image
            mode: 'auto' (auto-remove section), 'manual' (use rect),
                  'content' (crop to content), 'polygon', 'freehand'
            rect: Rectangle for manual mode
            points: Points for polygon/freehand mode
            keep_side: For auto mode: 'auto', 'left', or 'right'

        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load image
        img = Image.open(input_path)
        img_array = np.array(img)

        result = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'mode': mode,
            'success': False
        }

        try:
            if mode == 'auto':
                cropped, metadata = self.auto_remove_section(img_array, keep_side=keep_side)
                result.update(metadata)
            elif mode == 'manual':
                if rect is None:
                    raise ValueError("Rectangle required for manual mode")
                cropped = self.manual_crop(img_array, rect)
                result['rect'] = rect
            elif mode == 'content':
                cropped, metadata = self.crop_to_content(img_array)
                result.update(metadata)
            elif mode == 'polygon':
                if points is None:
                    raise ValueError("Points required for polygon mode")
                cropped, metadata = self.polygon_crop(img_array, points)
                result.update(metadata)
            elif mode == 'freehand':
                if points is None:
                    raise ValueError("Points required for freehand mode")
                cropped, metadata = self.freehand_crop(img_array, points)
                result.update(metadata)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save cropped image
            cropped_img = Image.fromarray(cropped)
            cropped_img.save(output_path)

            result['success'] = True
            result['output_size'] = (cropped.shape[1], cropped.shape[0])

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to process {input_path}: {e}")

        return result


def create_crop_processor() -> CropProcessor:
    """Factory function to create a CropProcessor instance."""
    return CropProcessor()
