"""OCR Service abstraction for neural document processing."""

import importlib
import importlib.util
import os
import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class OCRService(ABC):
    """Abstract base class for OCR services."""
    
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness from image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Layout-aware extracted text as markdown
        """
        pass


class NanonetsOCRService(OCRService):
    """Nanonets OCR implementation using NanonetsDocumentProcessor."""
    
    def __init__(self):
        """Initialize the service."""
        from .nanonets_processor import NanonetsDocumentProcessor
        self._processor = NanonetsDocumentProcessor()
        logger.info("NanonetsOCRService initialized")
    
    @property
    def model(self):
        """Get the Nanonets model."""
        return self._processor.model
    
    @property
    def processor(self):
        """Get the Nanonets processor."""
        return self._processor.processor
    
    @property
    def tokenizer(self):
        """Get the Nanonets tokenizer."""
        return self._processor.tokenizer
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Nanonets OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text(image_path)
                logger.info(f"Extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Nanonets OCR extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Nanonets OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text_with_layout(image_path)
                logger.info(f"Layout-aware extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Nanonets OCR layout-aware extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Nanonets OCR layout-aware extraction failed: {e}")
            return ""


class NeuralOCRService(OCRService):
    """Neural OCR implementation using docling's pre-trained models."""
    
    def __init__(self):
        """Initialize the service."""
        from .neural_document_processor import NeuralDocumentProcessor
        self._processor = NeuralDocumentProcessor()
        logger.info("NeuralOCRService initialized")
    
    def extract_text(self, image_path: str) -> str:
        """Extract text using Neural OCR (docling models)."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text(image_path)
                logger.info(f"Extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Neural OCR extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Neural OCR extraction failed: {e}")
            return ""
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Neural OCR."""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""
            
            # Check if file is readable
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    logger.info(f"Image loaded successfully: {img.size} {img.mode}")
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return ""
            
            try:
                text = self._processor.extract_text_with_layout(image_path)
                logger.info(f"Layout-aware extracted text length: {len(text)}")
                return text.strip()
            except Exception as e:
                logger.error(f"Neural OCR layout-aware extraction failed: {e}")
                return ""
                
        except Exception as e:
            logger.error(f"Neural OCR layout-aware extraction failed: {e}")
            return ""


class GoogleVisionOCRService(OCRService):
    """OCR implementation powered by Google Cloud Vision API."""

    def __init__(self):
        """Initialize Google Vision client and validate dependencies."""
        if not _is_google_vision_available():
            raise ImportError(
                "google-cloud-vision is required to use the Google OCR provider. "
                "Install it with `pip install docstrange[google-ocr]` or "
                "`pip install google-cloud-vision`."
            )

        vision_module = importlib.import_module("google.cloud.vision")
        self._vision = vision_module
        self._client = vision_module.ImageAnnotatorClient()

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            logger.info(
                "GoogleVisionOCRService initialized using credentials at %s",
                credentials_path,
            )
        else:
            logger.info(
                "GoogleVisionOCRService initialized with default Google credentials"
            )

    def extract_text(self, image_path: str) -> str:
        """Extract text using Google Vision OCR."""
        try:
            image = self._load_image(image_path)
            if image is None:
                return ""

            response = self._client.document_text_detection(image=image)
            _raise_for_google_error(response)

            annotation = response.full_text_annotation
            if annotation and annotation.text:
                text = annotation.text.strip()
                logger.info("Google Vision extracted text length: %d", len(text))
                return text
            return ""
        except Exception as e:
            logger.error(f"Google Vision OCR extraction failed: {e}")
            return ""

    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract layout-aware text using Google Vision OCR."""
        try:
            image = self._load_image(image_path)
            if image is None:
                return ""

            response = self._client.document_text_detection(image=image)
            _raise_for_google_error(response)

            annotation = response.full_text_annotation
            if not annotation:
                return ""

            layout_lines: List[str] = []
            pages = getattr(annotation, "pages", [])

            for index, page in enumerate(pages, start=1):
                layout_lines.append(f"### Page {index}")
                block_lines: List[str] = []

                for block in getattr(page, "blocks", []):
                    paragraph_lines: List[str] = []
                    for paragraph in getattr(block, "paragraphs", []):
                        words: List[str] = []
                        for word in getattr(paragraph, "words", []):
                            symbols = getattr(word, "symbols", [])
                            symbol_text = "".join(getattr(symbol, "text", "") for symbol in symbols)
                            if symbol_text:
                                words.append(symbol_text)
                        paragraph_text = " ".join(words).strip()
                        if paragraph_text:
                            paragraph_lines.append(paragraph_text)

                    block_text = "\n".join(paragraph_lines).strip()
                    if block_text:
                        block_lines.append(block_text)

                if block_lines:
                    layout_lines.append("\n\n".join(block_lines))

            if layout_lines:
                markdown_output = "\n\n".join(line for line in layout_lines if line).strip()
                if markdown_output:
                    logger.info(
                        "Google Vision layout-aware output length: %d",
                        len(markdown_output),
                    )
                    return markdown_output

            # Fallback to plain text if layout conversion produced nothing
            fallback_text = annotation.text.strip() if annotation and annotation.text else ""
            return fallback_text
        except Exception as e:
            logger.error(f"Google Vision OCR layout-aware extraction failed: {e}")
            return ""

    def _load_image(self, image_path: str):
        """Load image bytes for Google Vision API."""
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None

        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            return self._vision.Image(content=content)
        except Exception as e:
            logger.error(f"Failed to load image for Google Vision OCR: {e}")
            return None


class OCRServiceFactory:
    """Factory for creating OCR services based on configuration."""
    
    @staticmethod
    def create_service(provider: str = None) -> OCRService:
        """Create OCR service based on provider configuration.
        
        Args:
            provider: OCR provider name (defaults to config)
            
        Returns:
            OCRService instance
        """
        from docstrange.config import InternalConfig
        
        if provider is None:
            provider = getattr(InternalConfig, 'ocr_provider', 'nanonets')

        normalized = provider.lower()

        if normalized in {'nanonets', 'nanonets-ocr'}:
            return NanonetsOCRService()
        if normalized in {'neural', 'docling'}:
            return NeuralOCRService()
        if normalized in {'google', 'google-vision', 'google_vision'}:
            return GoogleVisionOCRService()

        raise ValueError(f"Unsupported OCR provider: {provider}")

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available OCR providers.

        Returns:
            List of available provider names
        """
        providers = ['nanonets', 'neural']
        if _is_google_vision_available():
            providers.append('google')
        return providers


def _is_google_vision_available() -> bool:
    """Check whether google-cloud-vision is installed."""

    return importlib.util.find_spec("google.cloud.vision") is not None


def _raise_for_google_error(response) -> None:
    """Raise a RuntimeError if the Google Vision response contains an error."""

    error = getattr(response, "error", None)
    message = getattr(error, "message", "") if error else ""
    if message:
        raise RuntimeError(f"Google Vision API error: {message}")
