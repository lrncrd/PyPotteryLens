"""
AI-Powered Metadata Extractor for PyPotteryLens
Uses Claude (Anthropic) or GPT (OpenAI) for intelligent metadata extraction from pottery images.
"""

import base64
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path


@dataclass
class AIExtractionConfig:
    """Configuration for AI extraction."""
    provider: str = "anthropic"  # 'anthropic', 'openai', 'gemini', 'lmstudio', 'ollama', 'deepseek', 'together'
    model: str = ""  # Will be set based on provider
    max_tokens: int = 2048
    temperature: float = 0.1
    api_key: str = ""
    base_url: str = ""  # For LM Studio/Ollama custom endpoints

    def __post_init__(self):
        if not self.model:
            if self.provider == "anthropic":
                self.model = "claude-sonnet-4-20250514"
            elif self.provider == "openai":
                self.model = "gpt-4.1-2025-04-14"
            elif self.provider == "gemini":
                self.model = "gemini-2.0-flash"
            elif self.provider == "lmstudio":
                self.model = "local-model"  # LM Studio uses whatever model is loaded
            elif self.provider == "ollama":
                self.model = "llava"  # Default vision model for Ollama
            elif self.provider == "deepseek":
                self.model = "deepseek-chat"  # Default DeepSeek model
            elif self.provider == "together":
                self.model = "meta-llama/Llama-Vision-Free"  # Free vision model on Together AI

        # Set default base URLs for local providers
        if not self.base_url:
            if self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434"
            elif self.provider == "deepseek":
                self.base_url = "https://api.deepseek.com"
            elif self.provider == "together":
                self.base_url = "https://api.together.xyz/v1"


@dataclass
class ExtractionResult:
    """Result of metadata extraction for a single image."""
    success: bool = False
    figure_number: str = ""
    pottery_id: str = ""
    period: str = ""
    original_period: str = ""  # Original language
    original_language: str = ""
    description: str = ""
    dimensions_cm: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    error: str = ""
    raw_response: str = ""


@dataclass
class DocumentStructure:
    """
    Represents the analyzed structure of an archaeological document.

    This is used to pre-analyze the entire PDF and build mappings
    between figure/plate numbers and archaeological periods.
    """
    # Maps figure/plate refs to periods: {"Tafel 3": "Umm an-Nar", "Abb. 174": "Wadi Suq"}
    tafel_period_map: Dict[str, str] = field(default_factory=dict)

    # Maps context ranges: {"Tomb 155": {"start": 1, "end": 8, "period": "Umm an-Nar"}}
    figure_ranges: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Maps catalog IDs to periods: {"BAT10A-0177": "Umm an-Nar"}
    catalog_entries: Dict[str, str] = field(default_factory=dict)

    # Detected primary language of the document
    language: str = ""

    # Whether the structure has been analyzed
    analyzed: bool = False

    # Any error during analysis
    error: str = ""

    def lookup_period(self, figure_ref: str = "", pottery_id: str = "") -> str:
        """
        Look up period from document structure.

        Args:
            figure_ref: Figure reference like "Tafel 3", "Abb. 174"
            pottery_id: Pottery catalog ID like "BAT10A-0177"

        Returns:
            Period string or empty string if not found
        """
        # Try direct figure/tafel lookup first
        if figure_ref:
            # Normalize the reference
            normalized = figure_ref.strip()
            if normalized in self.tafel_period_map:
                return self.tafel_period_map[normalized]

            # Try partial matching
            for key, period in self.tafel_period_map.items():
                if key.lower() in normalized.lower() or normalized.lower() in key.lower():
                    return period

            # Try to extract number and check figure ranges
            numbers = re.findall(r'\d+', normalized)
            if numbers:
                num = int(numbers[0])
                for context, info in self.figure_ranges.items():
                    if info.get('start', 0) <= num <= info.get('end', 0):
                        return info.get('period', '')

        # Try catalog ID lookup
        if pottery_id:
            normalized_id = pottery_id.strip()
            if normalized_id in self.catalog_entries:
                return self.catalog_entries[normalized_id]

            # Try partial matching
            for key, period in self.catalog_entries.items():
                if key in normalized_id or normalized_id in key:
                    return period

        return ""


class DocumentStructureAnalyzer:
    """
    Analyzes entire PDF documents to extract structure and period mappings.

    This performs a comprehensive analysis of the document to understand:
    - Which figures/plates belong to which archaeological period
    - Catalog entries and their period associations
    - Document structure (chapters, sections about specific periods)
    """

    STRUCTURE_PROMPT = """You are analyzing an archaeological publication to understand its structure and extract period mappings.

**TASK:** Extract comprehensive mappings between Figure/Plate/Tafel numbers and archaeological periods.

**WHAT TO LOOK FOR:**

1. **Chapter/Section Headers** mentioning periods:
   - "Chapter 3: Die Keramik des Grabes 155" (then find what period Tomb 155 belongs to)
   - "3.2 Umm an-Nar Period Ceramics"
   - "Die Wadi Suq-zeitliche Keramik"

2. **Figure Captions** with period attributions:
   - "Tafel 1-8: Keramik aus Grab 155, Umm an-Nar-Zeit"
   - "Abb. 174: Schalen der Wadi Suq-Zeit"
   - "Fig. 3: Iron Age pottery from Tomb 156"

3. **Table of Contents** or figure lists linking figures to periods

4. **Range Statements**:
   - "Tafeln 3-12 zeigen Keramik des Grabes 155" → find Tomb 155's period
   - "Plates 1-8 show Umm an-Nar material"

5. **Catalog Entries** with periods:
   - Tables listing pottery IDs with their periods
   - Entries like "BAT10A-0177: Umm an-Nar"

**LANGUAGES TO RECOGNIZE:**
- German: "Tafel", "Abb.", "Abbildung", "Umm an-Nar-Zeit", "Wadi Suq-zeitlich", "Späte Bronzezeit", "Eisenzeit", "Grab" (tomb)
- English: "Plate", "Fig.", "Figure", "Umm an-Nar period", "Wadi Suq", "Late Bronze Age", "Iron Age"
- Italian: "Tavola", "Tav.", "Figura", "Età del Bronzo", "Età del Ferro"

**PERIODS TO IDENTIFY:**
Arabian Peninsula chronology:
- Hafit (3100-2700 BCE)
- Umm an-Nar (2700-2000 BCE)
- Wadi Suq (2000-1600 BCE)
- Late Bronze Age (1600-1300 BCE)
- Iron Age I (1300-1100 BCE), II (1100-600 BCE), III (600-300 BCE)
- Samad/Late Pre-Islamic (300 BCE - 600 CE)
- Islamic (600 CE onwards)

**CRITICAL:** Look for statements that define which figures belong to which tomb/grave, then find what period that tomb belongs to. For example:
- "Tafel 3-8: Die Keramik des Grabes 155" + "Grab 155 datiert in die Umm an-Nar-Zeit"
- → Therefore Tafel 3, 4, 5, 6, 7, 8 are all Umm an-Nar period

Return ONLY valid JSON in this exact format:
{
    "tafel_period_map": {
        "Tafel 1": "Umm an-Nar",
        "Tafel 2": "Umm an-Nar",
        "Tafel 3": "Umm an-Nar",
        "Abb. 174": "Wadi Suq"
    },
    "figure_ranges": {
        "Tomb 155": {"start": 3, "end": 8, "period": "Umm an-Nar"},
        "Tomb 156": {"start": 9, "end": 13, "period": "Umm an-Nar"}
    },
    "catalog_entries": {
        "BAT10A-0177": "Umm an-Nar",
        "SU 53": "Wadi Suq"
    },
    "language": "de"
}"""

    def __init__(self, extractor: 'AIMetadataExtractor'):
        """
        Initialize with an AI extractor.

        Args:
            extractor: The AI extractor (Claude or OpenAI) to use for analysis
        """
        self.extractor = extractor

    def analyze_document(self, pdf_text: str, max_chunk_size: int = 30000) -> DocumentStructure:
        """
        Analyze the full PDF text to extract document structure.

        Args:
            pdf_text: Complete text content of the PDF
            max_chunk_size: Maximum characters to send to AI at once

        Returns:
            DocumentStructure with extracted mappings
        """
        result = DocumentStructure()

        try:
            # If text is too long, we need to be strategic about what we send
            if len(pdf_text) > max_chunk_size:
                # Extract key sections: table of contents, figure lists, chapters about periods
                important_sections = self._extract_important_sections(pdf_text, max_chunk_size)
                text_to_analyze = important_sections
            else:
                text_to_analyze = pdf_text

            # Use the extractor's underlying API to analyze the document
            if isinstance(self.extractor, ClaudeExtractor):
                response = self._analyze_with_claude(text_to_analyze)
            elif isinstance(self.extractor, (OpenAIExtractor, LMStudioExtractor, DeepSeekExtractor)):
                response = self._analyze_with_openai_compatible(text_to_analyze)
            elif isinstance(self.extractor, GeminiExtractor):
                response = self._analyze_with_gemini(text_to_analyze)
            elif isinstance(self.extractor, OllamaExtractor):
                response = self._analyze_with_ollama(text_to_analyze)
            else:
                result.error = "Unknown extractor type"
                return result

            # Parse the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                result.tafel_period_map = data.get('tafel_period_map', {})
                result.figure_ranges = data.get('figure_ranges', {})
                result.catalog_entries = data.get('catalog_entries', {})
                result.language = data.get('language', '')
                result.analyzed = True
            else:
                result.error = "No valid JSON in AI response"

        except json.JSONDecodeError as e:
            result.error = f"JSON parse error: {e}"
        except Exception as e:
            result.error = f"Analysis error: {e}"

        return result

    def _extract_important_sections(self, pdf_text: str, max_size: int) -> str:
        """
        Extract the most important sections from a long PDF.

        Focuses on: table of contents, figure lists, period descriptions,
        and sections mentioning Tafel/Figure/plates.
        """
        lines = pdf_text.split('\n')
        important_lines = []

        # Keywords that indicate important sections
        keywords = [
            'tafel', 'abb.', 'abbildung', 'fig', 'figure', 'plate', 'tav', 'tavola',
            'umm an-nar', 'wadi suq', 'bronze', 'iron', 'eisenzeit', 'bronzezeit',
            'grab ', 'tomb', 'keramik', 'ceramic', 'pottery',
            'inhaltsverzeichnis', 'contents', 'indice',
            'periode', 'period', 'zeit', 'età',
            'kapitel', 'chapter', 'capitolo'
        ]

        # Collect lines containing keywords with context
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                # Add surrounding context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                for j in range(start, end):
                    if lines[j] not in important_lines:
                        important_lines.append(lines[j])

        # Also get first and last sections (often have structure info)
        first_section = '\n'.join(lines[:200])
        last_section = '\n'.join(lines[-200:])

        combined = first_section + '\n\n--- EXTRACTED SECTIONS ---\n\n' + '\n'.join(important_lines) + '\n\n--- END SECTIONS ---\n\n' + last_section

        # Truncate if still too long
        if len(combined) > max_size:
            combined = combined[:max_size]

        return combined

    def _analyze_with_claude(self, text: str) -> str:
        """Analyze document structure using Claude."""
        client = self.extractor._get_client()

        response = client.messages.create(
            model=self.extractor.config.model,
            max_tokens=4096,
            system=self.STRUCTURE_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyze this archaeological document and extract the structure:\n\n{text}"
            }]
        )

        return response.content[0].text

    def _analyze_with_openai_compatible(self, text: str) -> str:
        """Analyze document structure using OpenAI-compatible API (OpenAI, LM Studio, DeepSeek)."""
        client = self.extractor._get_client()

        response = client.chat.completions.create(
            model=self.extractor.config.model,
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {"role": "system", "content": self.STRUCTURE_PROMPT},
                {"role": "user", "content": f"Analyze this archaeological document and extract the structure:\n\n{text}"}
            ]
        )

        return response.choices[0].message.content

    def _analyze_with_gemini(self, text: str) -> str:
        """Analyze document structure using Gemini."""
        import google.generativeai as genai

        model = self.extractor._get_client()

        prompt = self.STRUCTURE_PROMPT + f"\n\nAnalyze this archaeological document and extract the structure:\n\n{text}"

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=4096,
                temperature=0.1
            )
        )

        return response.text

    def _analyze_with_ollama(self, text: str) -> str:
        """Analyze document structure using Ollama."""
        prompt = self.STRUCTURE_PROMPT + f"\n\nAnalyze this archaeological document and extract the structure:\n\n{text}"

        data = {
            "model": self.extractor.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096
            }
        }

        response = self.extractor._make_request("/api/generate", data)
        return response.get("response", "")


class AIMetadataExtractor(ABC):
    """Abstract base class for AI metadata extractors."""

    SYSTEM_PROMPT = """You are an expert archaeological pottery analyst specializing in Near Eastern and Arabian Peninsula archaeology.

Your task is to extract metadata from pottery images by cross-referencing the image with the provided PDF document context.

**EXTRACTION PRIORITIES:**

1. **Figure/Table Number**: Look for references like "Abb.", "Fig.", "Tav.", "Tab.", "Plate" followed by numbers in the PDF context. Match the image to its figure/table reference.

2. **Pottery ID / Inventory Number**: Extract catalog numbers like "BAT10A-0177", "M5-12", "SU 53", "GRO-15". These often appear in tables or captions.

3. **Archaeological Period**: THIS IS CRITICAL. Look in the PDF context for period attributions. Common periods include:
   - Arabian Peninsula: Umm an-Nar, Wadi Suq, Late Bronze Age (LBA), Iron Age I/II/III, Samad, Islamic
   - Near East: Early/Middle/Late Bronze Age, Iron Age, Hellenistic, Roman, Byzantine
   - Look for German terms: Späte Bronzezeit (Late Bronze Age), Eisenzeit (Iron Age)
   - The period is often mentioned in figure captions or nearby text tables

**KEY INSTRUCTIONS:**
- ALWAYS check the PDF context first for period information - the image alone rarely shows the period
- Look for tables in the PDF that list pottery IDs with their periods
- Figure captions often specify the period (e.g., "Abb. 174: Schalen... Wadi Suq-zeitlich")
- If the PDF mentions a pottery ID, find its associated period in nearby text or tables
- Return empty string if you cannot confidently determine a value

Respond ONLY with valid JSON:
{
    "figure_number": "string or empty",
    "pottery_id": "string or empty",
    "period": "archaeological period in English",
    "original_period": "original text from document",
    "original_language": "de/en/it/fr/es/ar",
    "scale_cm": number or null,
    "confidence": 0.0-1.0
}"""

    def __init__(self, config: AIExtractionConfig):
        """Initialize the extractor with configuration."""
        self.config = config

    @abstractmethod
    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """
        Extract metadata from a single image.

        Args:
            image_base64: Base64 encoded image data
            context: Optional additional context (e.g., PDF text)
            media_type: MIME type of the image (e.g., 'image/png', 'image/jpeg')
            document_structure: Optional pre-analyzed document structure for period lookup

        Returns:
            ExtractionResult with extracted metadata
        """
        pass

    @abstractmethod
    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """
        Extract period mappings from PDF text.

        Args:
            pdf_text: Full text content of reference PDF

        Returns:
            Dictionary mapping pottery IDs to periods
        """
        pass

    def _parse_response(self, response_text: str) -> ExtractionResult:
        """Parse AI response into ExtractionResult."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return ExtractionResult(
                    success=False,
                    error="No JSON found in response",
                    raw_response=response_text
                )

            data = json.loads(json_match.group())

            return ExtractionResult(
                success=True,
                figure_number=data.get('figure_number', ''),
                pottery_id=data.get('pottery_id', ''),
                period=data.get('period', ''),
                original_period=data.get('original_period', ''),
                original_language=data.get('original_language', ''),
                description=data.get('description', ''),
                dimensions_cm={'scale': data.get('scale_cm')} if data.get('scale_cm') else {},
                confidence=float(data.get('confidence', 0.5)),
                raw_response=response_text
            )

        except json.JSONDecodeError as e:
            return ExtractionResult(
                success=False,
                error=f"JSON parse error: {e}",
                raw_response=response_text
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                error=str(e),
                raw_response=response_text
            )


class ClaudeExtractor(AIMetadataExtractor):
    """Metadata extractor using Anthropic's Claude API."""

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)
        self.client = None

    def _get_client(self):
        """Lazy load the Anthropic client."""
        if self.client is None:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self.client

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using Claude's vision capabilities."""
        try:
            client = self._get_client()

            # Build user message with image and optional context
            user_content = []

            # Add image
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64
                }
            })

            # Build prompt with document structure if available
            prompt = """Analyze this archaeological pottery image and extract metadata.

IMPORTANT: Use the PDF context below to find the archaeological period for this pottery.
- Look for figure/table references (Abb., Fig., Tav.) that match this image
- Find the period attribution in the caption or nearby text
- Check any tables that list pottery IDs with their periods
- The period information is in the document, not in the image itself"""

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

                if document_structure.catalog_entries:
                    prompt += "\nCatalog ID to Period mappings:\n"
                    for cat_id, period in list(document_structure.catalog_entries.items())[:20]:
                        prompt += f"  {cat_id} -> {period}\n"

            if context:
                prompt += f"\n\n=== PDF DOCUMENT CONTEXT ===\n{context}"

            user_content.append({
                "type": "text",
                "text": prompt
            })

            # Make API call
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": user_content
                }]
            )

            # Parse response
            response_text = response.content[0].text
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period  # Could be refined

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Claude API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using Claude."""
        try:
            client = self._get_client()

            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            response = client.messages.create(
                model=self.config.model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = response.content[0].text

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


class OpenAIExtractor(AIMetadataExtractor):
    """Metadata extractor using OpenAI's GPT API."""

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)
        self.client = None

    def _get_client(self):
        """Lazy load the OpenAI client."""
        if self.client is None:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self.client

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using GPT's vision capabilities."""
        try:
            client = self._get_client()

            # Build prompt with document structure if available
            prompt = "Analyze this archaeological pottery image and extract metadata."

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

            if context:
                prompt += f"\n\nAdditional context from the source document:\n{context[:2000]}"

            user_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]

            # Make API call
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ]
            )

            # Parse response
            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"OpenAI API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using GPT."""
        try:
            client = self._get_client()

            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=4096,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = response.choices[0].message.content

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


class GeminiExtractor(AIMetadataExtractor):
    """Metadata extractor using Google's Gemini API."""

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)
        self.client = None

    def _get_client(self):
        """Lazy load the Gemini client."""
        if self.client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                self.client = genai.GenerativeModel(self.config.model)
            except ImportError:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        return self.client

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using Gemini's vision capabilities."""
        try:
            import google.generativeai as genai
            from PIL import Image
            import io

            model = self._get_client()

            # Convert base64 to PIL Image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            # Build prompt
            prompt = self.SYSTEM_PROMPT + "\n\nAnalyze this archaeological pottery image and extract metadata."

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

            if context:
                prompt += f"\n\n=== PDF DOCUMENT CONTEXT ===\n{context[:8000]}"

            # Make API call
            response = model.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            )

            # Parse response
            response_text = response.text
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Gemini API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using Gemini."""
        try:
            import google.generativeai as genai

            model = self._get_client()

            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=4096,
                    temperature=0.1
                )
            )

            response_text = response.text

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


class LMStudioExtractor(AIMetadataExtractor):
    """
    Metadata extractor using LM Studio's OpenAI-compatible API.
    LM Studio runs local models and exposes them via an OpenAI-compatible endpoint.
    """

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)
        self.client = None

    def _get_client(self):
        """Lazy load the OpenAI client configured for LM Studio."""
        if self.client is None:
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url=self.config.base_url,
                    api_key="lm-studio"  # LM Studio doesn't require a real API key
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self.client

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using LM Studio's vision capabilities (if model supports it)."""
        try:
            client = self._get_client()

            # Build prompt
            prompt = "Analyze this archaeological pottery image and extract metadata."

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

            if context:
                prompt += f"\n\nAdditional context from the source document:\n{context[:2000]}"

            # Try vision API first (for multimodal models like LLaVA)
            try:
                user_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]

                response = client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                )
            except Exception as vision_error:
                # Fallback to text-only if vision not supported
                print(f"Vision not supported, falling back to text-only: {vision_error}")
                response = client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt + "\n\n[Image analysis not available - please infer from context]"}
                    ]
                )

            # Parse response
            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"LM Studio API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using LM Studio."""
        try:
            client = self._get_client()

            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=4096,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = response.choices[0].message.content

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


class OllamaExtractor(AIMetadataExtractor):
    """
    Metadata extractor using Ollama's local LLM API.
    Ollama runs local models and provides a REST API.
    """

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)

    def _make_request(self, endpoint: str, data: dict) -> dict:
        """Make a request to Ollama API."""
        import urllib.request
        import urllib.error

        url = f"{self.config.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.config.base_url}: {e}")

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using Ollama's vision capabilities (llava, bakllava, etc.)."""
        try:
            # Build prompt
            prompt = self.SYSTEM_PROMPT + "\n\nAnalyze this archaeological pottery image and extract metadata."

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

            if context:
                prompt += f"\n\n=== PDF DOCUMENT CONTEXT ===\n{context[:4000]}"

            # Use Ollama's generate API with image
            data = {
                "model": self.config.model,
                "prompt": prompt,
                "images": [image_base64],  # Ollama expects base64 images in array
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }

            response = self._make_request("/api/generate", data)

            # Parse response
            response_text = response.get("response", "")
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Ollama API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using Ollama."""
        try:
            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            data = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 4096
                }
            }

            response = self._make_request("/api/generate", data)
            response_text = response.get("response", "")

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


class DeepSeekExtractor(AIMetadataExtractor):
    """
    Metadata extractor using DeepSeek's API.
    DeepSeek provides an OpenAI-compatible API.
    """

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)
        self.client = None

    def _get_client(self):
        """Lazy load the OpenAI client configured for DeepSeek."""
        if self.client is None:
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self.client

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using DeepSeek (text-based, uses context heavily)."""
        try:
            client = self._get_client()

            # Build prompt - DeepSeek is text-focused so we rely on context
            prompt = "Analyze the following archaeological pottery information and extract metadata."

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

                if document_structure.catalog_entries:
                    prompt += "\nCatalog ID to Period mappings:\n"
                    for cat_id, period in list(document_structure.catalog_entries.items())[:20]:
                        prompt += f"  {cat_id} -> {period}\n"

            if context:
                prompt += f"\n\n=== PDF DOCUMENT CONTEXT ===\n{context[:6000]}"

            prompt += "\n\nBased on the above context, extract metadata for the pottery item."

            # DeepSeek uses chat completions API
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response
            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"DeepSeek API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using DeepSeek."""
        try:
            client = self._get_client()

            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=4096,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = response.choices[0].message.content

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


class TogetherExtractor(AIMetadataExtractor):
    """
    Metadata extractor using Together AI's API.
    Together AI hosts open-source models including vision models like LLaVA.
    Uses OpenAI-compatible API at https://api.together.xyz/v1
    """

    def __init__(self, config: AIExtractionConfig):
        super().__init__(config)
        self.client = None

    def _get_client(self):
        """Lazy load the OpenAI client configured for Together AI."""
        if self.client is None:
            try:
                import openai
                self.client = openai.OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key
                )
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self.client

    def extract_metadata(self, image_base64: str, context: str = "",
                        media_type: str = "image/png",
                        document_structure: Optional['DocumentStructure'] = None) -> ExtractionResult:
        """Extract metadata using Together AI with vision models."""
        try:
            client = self._get_client()

            # Build prompt with document structure if available
            prompt = "Analyze this archaeological pottery image and extract metadata."

            # Add document structure mappings if available
            if document_structure and document_structure.analyzed:
                prompt += "\n\n=== PRE-ANALYZED DOCUMENT STRUCTURE ===\n"
                prompt += "Use these mappings to determine the period:\n"

                if document_structure.tafel_period_map:
                    prompt += "\nFigure/Tafel to Period mappings:\n"
                    for tafel, period in list(document_structure.tafel_period_map.items())[:20]:
                        prompt += f"  {tafel} -> {period}\n"

                if document_structure.figure_ranges:
                    prompt += "\nFigure ranges by context:\n"
                    for ctx, info in document_structure.figure_ranges.items():
                        prompt += f"  {ctx}: Figures {info.get('start')}-{info.get('end')} = {info.get('period')}\n"

                if document_structure.catalog_entries:
                    prompt += "\nCatalog ID to Period mappings:\n"
                    for cat_id, period in list(document_structure.catalog_entries.items())[:20]:
                        prompt += f"  {cat_id} -> {period}\n"

            if context:
                prompt += f"\n\n=== PDF DOCUMENT CONTEXT ===\n{context[:6000]}"

            # Together AI vision models use OpenAI-compatible format
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=messages
            )

            # Parse response
            response_text = response.choices[0].message.content
            result = self._parse_response(response_text)

            # If AI didn't find period but we have document structure, try lookup
            if result.success and not result.period and document_structure:
                looked_up_period = document_structure.lookup_period(
                    figure_ref=result.figure_number,
                    pottery_id=result.pottery_id
                )
                if looked_up_period:
                    result.period = looked_up_period
                    result.original_period = looked_up_period

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error=f"Together AI API error: {str(e)}"
            )

    def extract_periods_from_pdf(self, pdf_text: str) -> Dict[str, str]:
        """Extract period mappings from PDF text using Together AI."""
        try:
            client = self._get_client()

            prompt = f"""Analyze this archaeological document text and extract a mapping of pottery IDs to their chronological periods.

Document text (truncated):
{pdf_text[:8000]}

Return a JSON object where keys are pottery IDs (e.g., "M5-12", "SU 53") and values are their periods in English.
Only include entries where you can clearly identify both the ID and period.

Response format:
{{"M5-12": "Late Bronze Age", "SU 53": "Iron Age II", ...}}"""

            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=4096,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            response_text = response.choices[0].message.content

            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())

            return {}

        except Exception as e:
            print(f"Error extracting periods: {e}")
            return {}


def get_extractor(provider: str, api_key: str = "", base_url: str = "", model: str = "") -> AIMetadataExtractor:
    """
    Factory function to get the appropriate extractor.

    Args:
        provider: 'anthropic', 'openai', 'gemini', 'lmstudio', or 'ollama'
        api_key: API key for the provider (not needed for local providers)
        base_url: Custom base URL (for lmstudio/ollama)
        model: Optional model override

    Returns:
        Configured AIMetadataExtractor instance
    """
    config = AIExtractionConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url
    )

    # Override model if provided
    if model:
        config.model = model

    if provider == "anthropic":
        return ClaudeExtractor(config)
    elif provider == "openai":
        return OpenAIExtractor(config)
    elif provider == "gemini":
        return GeminiExtractor(config)
    elif provider == "lmstudio":
        return LMStudioExtractor(config)
    elif provider == "ollama":
        return OllamaExtractor(config)
    elif provider == "deepseek":
        return DeepSeekExtractor(config)
    elif provider == "together":
        return TogetherExtractor(config)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: anthropic, openai, gemini, lmstudio, ollama, deepseek, together")


def detect_image_media_type(image_path: str) -> str:
    """
    Detect the media type of an image file.

    Args:
        image_path: Path to image file

    Returns:
        Media type string (e.g., 'image/png', 'image/jpeg')
    """
    ext = Path(image_path).suffix.lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    return media_types.get(ext, 'image/png')


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class BatchExtractor:
    """Batch processor for extracting metadata from multiple images."""

    def __init__(self, extractor: AIMetadataExtractor, progress_callback=None):
        """
        Initialize batch extractor.

        Args:
            extractor: The AI extractor to use
            progress_callback: Optional callback(current, total, message)
        """
        self.extractor = extractor
        self.progress_callback = progress_callback

    def process_images(self, image_paths: List[str], pdf_context: str = "") -> List[Dict[str, Any]]:
        """
        Process multiple images and extract metadata.

        Args:
            image_paths: List of paths to image files
            pdf_context: Optional context from source PDF

        Returns:
            List of extraction results as dictionaries
        """
        results = []
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            if self.progress_callback:
                self.progress_callback(i, total, f"Processing {Path(image_path).name}")

            try:
                # Convert image to base64
                image_b64 = image_to_base64(image_path)

                # Extract metadata
                result = self.extractor.extract_metadata(image_b64, pdf_context)

                results.append({
                    'image': Path(image_path).name,
                    'success': result.success,
                    'figure_number': result.figure_number,
                    'pottery_id': result.pottery_id,
                    'period': result.period,
                    'original_period': result.original_period,
                    'original_language': result.original_language,
                    'description': result.description,
                    'confidence': result.confidence,
                    'error': result.error
                })

            except Exception as e:
                results.append({
                    'image': Path(image_path).name,
                    'success': False,
                    'error': str(e)
                })

        if self.progress_callback:
            self.progress_callback(total, total, "Complete")

        return results
