"""
Scale Bar Detector for PyPotteryLens
Automatically detects metric scale bars in archaeological pottery images.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import re


@dataclass
class ScaleBarConfig:
    """Configuration for scale bar detection."""
    min_length_ratio: float = 0.03  # Minimum scale bar length as ratio of image width
    max_length_ratio: float = 0.5   # Maximum scale bar length as ratio of image width
    min_line_length: int = 30       # Minimum line length in pixels for Hough
    max_line_gap: int = 15          # Maximum gap between line segments
    angle_tolerance: float = 5.0    # Degrees tolerance for horizontal lines
    typical_values: Tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)  # Common scale bar values in cm


@dataclass
class ScaleBarResult:
    """Result of scale bar detection."""
    detected: bool
    pixels: int = 0              # Length in pixels
    cm: float = 0.0              # Detected value in cm (if OCR found it)
    unit_text: str = ""          # Original text found (e.g., "5 cm", "10cm")
    confidence: float = 0.0      # Detection confidence (0-1)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # Bounding box of scale bar
    pixels_per_cm: float = 0.0   # Calculated pixels per cm


class ScaleBarDetector:
    """Detects scale bars in images using edge detection, line detection, and OCR."""

    def __init__(self, config: Optional[ScaleBarConfig] = None):
        """
        Initialize the scale bar detector.

        Args:
            config: Configuration options for detection.
        """
        self.config = config or ScaleBarConfig()
        self._ocr_reader = None

    def _get_ocr_reader(self):
        """Lazy load the OCR reader."""
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en', 'it', 'fr', 'de', 'es'], gpu=False)
            except ImportError:
                print("Warning: easyocr not available for scale bar text detection")
                return None
        return self._ocr_reader

    def detect(self, image: np.ndarray) -> ScaleBarResult:
        """
        Detect scale bar in image.

        Args:
            image: Input image as numpy array (BGR or grayscale).

        Returns:
            ScaleBarResult with detection information.
        """
        if image is None or image.size == 0:
            return ScaleBarResult(detected=False)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape[:2]

        # Step 1: Edge detection
        edges = self._detect_edges(gray)

        # Step 2: Find horizontal lines
        lines = self._find_horizontal_lines(edges, width)

        if not lines:
            return ScaleBarResult(detected=False)

        # Step 3: Find best scale bar candidate
        candidates = self._find_scale_bar_candidates(lines, width, height)

        if not candidates:
            return ScaleBarResult(detected=False)

        # Step 4: Try to read scale value with OCR
        best_candidate = max(candidates, key=lambda x: x['score'])

        # Get region around the line for OCR
        ocr_result = self._ocr_scale_region(image, best_candidate, height)

        # Build result
        result = ScaleBarResult(
            detected=True,
            pixels=best_candidate['length'],
            bbox=best_candidate['bbox'],
            confidence=min(1.0, best_candidate['score'] / 100),
            unit_text=ocr_result.get('text', ''),
            cm=ocr_result.get('value', 0.0)
        )

        if result.cm > 0:
            result.pixels_per_cm = result.pixels / result.cm

        return result

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better edge detection
        v = np.median(gray)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        edges = cv2.Canny(blurred, lower, upper)
        return edges

    def _find_horizontal_lines(self, edges: np.ndarray, width: int) -> List[Tuple]:
        """Find horizontal lines using Hough transform."""
        min_length = max(self.config.min_line_length, int(width * self.config.min_length_ratio))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=self.config.max_line_gap
        )

        if lines is None:
            return []

        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Check if horizontal (within tolerance)
            if angle <= self.config.angle_tolerance or (180 - angle) <= self.config.angle_tolerance:
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                horizontal_lines.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'length': int(length),
                    'angle': angle
                })

        return horizontal_lines

    def _find_scale_bar_candidates(self, lines: List[Dict], width: int, height: int) -> List[Dict]:
        """
        Find lines that are likely scale bars.

        Scale bars typically:
        - Are in the bottom portion of the image
        - Have a length between min and max ratio of image width
        - May have tick marks at endpoints
        """
        candidates = []

        min_length = int(width * self.config.min_length_ratio)
        max_length = int(width * self.config.max_length_ratio)

        for line in lines:
            length = line['length']

            # Check length constraints
            if length < min_length or length > max_length:
                continue

            # Calculate score based on various factors
            score = 0

            # Prefer lines in bottom half of image
            y_center = (line['y1'] + line['y2']) / 2
            if y_center > height * 0.5:
                score += 30
            if y_center > height * 0.7:
                score += 20

            # Prefer lines that are more horizontal
            score += max(0, 20 - line['angle'] * 4)

            # Prefer lines with length matching common scale values
            # (assuming typical scale bars are 50-200 pixels per cm)
            for typical_cm in self.config.typical_values:
                for ppcm in [50, 75, 100, 125, 150, 175, 200]:
                    expected_length = typical_cm * ppcm
                    if abs(length - expected_length) < expected_length * 0.1:
                        score += 10
                        break

            # Calculate bounding box
            x_min = min(line['x1'], line['x2'])
            x_max = max(line['x1'], line['x2'])
            y_min = min(line['y1'], line['y2']) - 20
            y_max = max(line['y1'], line['y2']) + 40  # Extra space below for text

            candidates.append({
                'length': length,
                'score': score,
                'line': line,
                'bbox': (max(0, x_min - 10), max(0, y_min), min(width, x_max + 10), min(height, y_max))
            })

        return candidates

    def _ocr_scale_region(self, image: np.ndarray, candidate: Dict, height: int) -> Dict:
        """
        Use OCR to read scale value near the detected line.

        Args:
            image: Original image
            candidate: Scale bar candidate
            height: Image height

        Returns:
            Dict with 'text' and 'value' keys
        """
        reader = self._get_ocr_reader()
        if reader is None:
            return {'text': '', 'value': 0.0}

        bbox = candidate['bbox']
        x1, y1, x2, y2 = bbox

        # Extend region below the line to capture text
        y2 = min(height, y2 + 30)

        # Crop region
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return {'text': '', 'value': 0.0}

        try:
            results = reader.readtext(region, detail=1)

            # Look for text containing numbers and unit indicators
            for (box, text, conf) in results:
                parsed = self._parse_scale_text(text)
                if parsed['value'] > 0:
                    return {'text': text, 'value': parsed['value']}

        except Exception as e:
            print(f"OCR error: {e}")

        return {'text': '', 'value': 0.0}

    def _parse_scale_text(self, text: str) -> Dict:
        """
        Parse scale bar text to extract numeric value.

        Examples:
            "5 cm" -> 5.0
            "10cm" -> 10.0
            "2.5 cm" -> 2.5
            "20 mm" -> 2.0 (converted to cm)
        """
        if not text:
            return {'value': 0.0}

        text = text.lower().strip()

        # Pattern for cm values
        cm_pattern = r'(\d+(?:[.,]\d+)?)\s*(?:cm|centimetr)'
        match = re.search(cm_pattern, text)
        if match:
            value = float(match.group(1).replace(',', '.'))
            return {'value': value}

        # Pattern for mm values (convert to cm)
        mm_pattern = r'(\d+(?:[.,]\d+)?)\s*(?:mm|millimetr)'
        match = re.search(mm_pattern, text)
        if match:
            value = float(match.group(1).replace(',', '.')) / 10.0
            return {'value': value}

        # Just a number (assume cm)
        num_pattern = r'^(\d+(?:[.,]\d+)?)$'
        match = re.search(num_pattern, text)
        if match:
            value = float(match.group(1).replace(',', '.'))
            if value in self.config.typical_values or value <= 50:
                return {'value': value}

        return {'value': 0.0}


def detect_scale_bar(image_path: str, config: Optional[ScaleBarConfig] = None) -> ScaleBarResult:
    """
    Convenience function to detect scale bar in an image file.

    Args:
        image_path: Path to image file
        config: Optional configuration

    Returns:
        ScaleBarResult
    """
    image = cv2.imread(image_path)
    if image is None:
        return ScaleBarResult(detected=False)

    detector = ScaleBarDetector(config)
    return detector.detect(image)


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scale_detector.py <image_path>")
        sys.exit(1)

    result = detect_scale_bar(sys.argv[1])
    print(f"Detected: {result.detected}")
    if result.detected:
        print(f"  Pixels: {result.pixels}")
        print(f"  CM: {result.cm}")
        print(f"  Text: {result.unit_text}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Pixels per CM: {result.pixels_per_cm:.2f}")
