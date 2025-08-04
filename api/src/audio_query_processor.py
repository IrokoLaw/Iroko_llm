"""
Classe orchestrateur AudioQueryProcessor pour gérer le workflow complet de traitement audio.

Ce module fournit l'orchestrateur principal qui coordonne la détection audio, le téléchargement,
la transcription et les opérations de nettoyage pour le pipeline de traitement audio-vers-texte.
"""

import logging
from typing import Optional
from config import AudioConfig
from src.audio_detector import AudioDetector
from src.audio_downloader import AudioDownloader, AudioDownloadError, AudioFormatError
from src.gemma_transcriber import GemmaTranscriber

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Exception de base pour les erreurs de traitement audio"""

    def __init__(self, message: str, error_type: str = "audio_processing_error"):
        super().__init__(message)
        self.error_type = error_type


class AudioURLError(AudioProcessingError):
    """Exception levée quand l'URL audio est invalide ou inaccessible"""

    def __init__(self, message: str):
        super().__init__(message, "invalid_url")


class AudioTranscriptionError(AudioProcessingError):
    """Exception levée quand la transcription audio échoue"""

    def __init__(self, message: str):
        super().__init__(message, "transcription_failed")


class AudioQueryProcessor:
    """
    Classe orchestrateur pour le workflow complet de traitement audio.

    Cette classe coordonne l'ensemble du processus depuis la détection d'URL audio
    jusqu'à la transcription, incluant la gestion d'erreurs et le nettoyage des ressources.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        detector: Optional[AudioDetector] = None,
        downloader: Optional[AudioDownloader] = None,
        transcriber: Optional[GemmaTranscriber] = None,
    ):
        """
        Initialize AudioQueryProcessor with components and configuration.

        Args:
            config: AudioConfig instance, uses default if None
            detector: AudioDetector instance, creates new if None
            downloader: AudioDownloader instance, creates new if None
            transcriber: GemmaTranscriber instance, creates new if None
        """
        self.config = config or AudioConfig()
        self.detector = detector or AudioDetector()
        self.downloader = downloader or AudioDownloader(
            max_size_mb=self.config.max_file_size_mb,
            timeout_seconds=self.config.download_timeout_seconds,
        )
        self.transcriber = transcriber or GemmaTranscriber(self.config)

    def process_query(self, query: str) -> str:
        """
        Process a query that may contain an audio URL or regular text.

        This method orchestrates the complete audio processing flow:
        1. Detect if query contains an audio URL
        2. If not audio, return original query unchanged
        3. If audio URL, validate accessibility
        4. Download the audio file
        5. Validate audio format
        6. Transcribe audio to text
        7. Clean up temporary resources
        8. Return transcribed text

        Args:
            query: The input query string (may be text or audio URL)

        Returns:
            str: Original query if text, or transcribed text if audio URL

        Raises:
            AudioURLError: If audio URL is invalid or inaccessible
            AudioProcessingError: If download, format validation, or transcription fails
        """
        if not query or not isinstance(query, str):
            return query or ""

        query = query.strip()
        if not query:
            return ""

        # Step 1: Check if query contains an audio URL
        if not self.detector.is_audio_url(query):
            logger.debug("Query is not an audio URL, returning as-is")
            return query

        logger.info("Audio URL detected, starting processing workflow")

        # Extract the audio URL
        audio_url = self.detector.extract_audio_url(query)
        if not audio_url:
            raise AudioURLError("Impossible d'extraire l'URL audio de la requête")

        # Step 2: Validate audio URL accessibility
        if not self.detector.validate_audio_url(audio_url):
            raise AudioURLError(
                f"URL audio inaccessible ou invalide: {audio_url}. "
                "Vérifiez que l'URL est correcte et accessible."
            )

        temp_file_path = None
        try:
            # Step 3: Download audio file
            logger.info(f"Downloading audio from URL: {audio_url}")
            temp_file_path = self.downloader.download_audio(audio_url)

            # Step 4: Validate audio format
            logger.info("Validating audio format")
            self.downloader.validate_audio_format(temp_file_path)

            # Step 5: Transcribe audio to text
            logger.info("Starting audio transcription")
            transcribed_text = self.transcriber.transcribe_audio(temp_file_path)

            if not transcribed_text or not transcribed_text.strip():
                raise AudioTranscriptionError(
                    "La transcription audio a produit un texte vide. "
                    "Vérifiez la qualité de l'enregistrement audio."
                )

            logger.info(
                f"Audio processing completed successfully. "
                f"Transcribed text length: {len(transcribed_text)} characters"
            )
            return transcribed_text

        except AudioDownloadError as e:
            logger.error(f"Audio download failed: {str(e)}")
            raise AudioProcessingError(str(e), "download_failed")

        except AudioFormatError as e:
            logger.error(f"Audio format validation failed: {str(e)}")
            raise AudioProcessingError(str(e), "unsupported_format")

        except AudioTranscriptionError:
            # Re-raise AudioTranscriptionError as-is
            raise

        except RuntimeError as e:
            # GemmaTranscriber raises RuntimeError for transcription failures
            logger.error(f"Audio transcription failed: {str(e)}")
            raise AudioTranscriptionError(str(e))

        except Exception as e:
            logger.error(f"Unexpected error during audio processing: {str(e)}")
            raise AudioProcessingError(
                f"Erreur inattendue lors du traitement audio: {str(e)}",
                "unexpected_error",
            )

        finally:
            # Step 6: Always cleanup temporary resources
            self._cleanup_resources(temp_file_path)

    def _cleanup_resources(self, temp_file_path: Optional[str]) -> None:
        """
        Clean up temporary resources created during audio processing.

        Args:
            temp_file_path: Path to temporary file to clean up, if any
        """
        if temp_file_path:
            try:
                self.downloader.cleanup_temp_file(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temporary file {temp_file_path}: {str(e)}"
                )

    def is_audio_query(self, query: str) -> bool:
        """
        Check if a query contains an audio URL without processing it.

        Args:
            query: The query string to check

        Returns:
            bool: True if query contains an audio URL, False otherwise
        """
        return self.detector.is_audio_url(query) if query else False

    def get_supported_formats(self) -> list:
        """
        Get list of supported audio formats.

        Returns:
            list: List of supported audio file extensions
        """
        return self.config.supported_formats.copy()

    def get_max_file_size_mb(self) -> int:
        """
        Get maximum allowed file size in megabytes.

        Returns:
            int: Maximum file size in MB
        """
        return self.config.max_file_size_mb
