"""Classe AudioDownloader pour gérer le téléchargement et la validation de fichiers audio"""

import os
import tempfile
import requests
import librosa
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioDownloadError(Exception):
    """Exception levée quand le téléchargement audio échoue"""

    pass


class AudioFormatError(Exception):
    """Exception levée quand le format audio est invalide"""

    pass


class AudioDownloader:
    """Classe pour télécharger et valider des fichiers audio depuis des URLs"""

    def __init__(self, max_size_mb: int = 50, timeout_seconds: int = 30):
        """
        Initialise AudioDownloader avec les paramètres de configuration

        Args:
            max_size_mb: Taille maximale du fichier en mégaoctets
            timeout_seconds: Timeout pour les requêtes de téléchargement en secondes
        """
        self.max_size_mb = max_size_mb
        self.timeout_seconds = timeout_seconds
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.supported_formats = ["mp3", "wav", "m4a", "ogg", "flac", "aac"]

    def download_audio(self, url: str) -> str:
        """
        Télécharge un fichier audio depuis une URL et le sauvegarde dans un fichier temporaire

        Args:
            url: URL du fichier audio à télécharger

        Returns:
            str: Chemin vers le fichier temporaire téléchargé

        Raises:
            AudioDownloadError: Si le téléchargement échoue ou si le fichier est trop volumineux
        """
        try:
            # Crée le répertoire temporaire s'il n'existe pas
            temp_dir = Path(tempfile.gettempdir()) / "iroko_audio"
            temp_dir.mkdir(exist_ok=True)

            # Démarre le téléchargement en streaming pour vérifier la taille
            response = requests.get(
                url,
                stream=True,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "IrokoAPI/1.0"},
            )
            response.raise_for_status()

            # Vérifie la longueur du contenu si fournie
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.max_size_bytes:
                raise AudioDownloadError(
                    f"Fichier trop volumineux: {int(content_length) / (1024*1024):.1f}MB. "
                    f"Limite: {self.max_size_mb}MB"
                )

            # Crée le fichier temporaire
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=self._get_file_extension(url), dir=temp_dir
            )

            # Téléchargement avec vérification de taille
            downloaded_size = 0
            chunk_size = 8192

            try:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_size_bytes:
                            temp_file.close()
                            os.unlink(temp_file.name)
                            raise AudioDownloadError(
                                f"Fichier trop volumineux: {downloaded_size / (1024*1024):.1f}MB. "
                                f"Limite: {self.max_size_mb}MB"
                            )
                        temp_file.write(chunk)

                temp_file.close()
                logger.info(
                    f"Audio téléchargé avec succès: {downloaded_size / (1024*1024):.1f}MB"
                )
                return temp_file.name

            except Exception as e:
                temp_file.close()
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise

        except requests.exceptions.Timeout:
            raise AudioDownloadError(
                f"Timeout lors du téléchargement (limite: {self.timeout_seconds}s)"
            )
        except requests.exceptions.RequestException as e:
            raise AudioDownloadError(f"Erreur lors du téléchargement: {str(e)}")
        except Exception as e:
            raise AudioDownloadError(
                f"Erreur inattendue lors du téléchargement: {str(e)}"
            )

    def validate_audio_format(self, file_path: str) -> bool:
        try:
            # Vérifie d'abord l'extension
            ext = Path(file_path).suffix.lower()[1:]
            if ext not in self.supported_formats:
                raise AudioFormatError(f"Format {ext} non supporté")
                
            # Si c'est un OGG, on convertit
            if ext == 'ogg':
                file_path = self.convert_to_mp3(file_path)
                
            # Validation avec librosa
            y, sr = librosa.load(file_path, sr=None, duration=1.0)
            return True
            
        except Exception as e:
            raise AudioFormatError(f"Validation échouée: {str(e)}")

    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary audio file

        Args:
            file_path: Path to the temporary file to delete
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Fichier temporaire supprimé: {file_path}")
        except Exception as e:
            logger.warning(
                f"Erreur lors de la suppression du fichier temporaire {file_path}: {str(e)}"
            )

    def _get_file_extension(self, url: str) -> str:
        """
        Extract file extension from URL

        Args:
            url: URL to extract extension from

        Returns:
            str: File extension with dot (e.g., '.mp3')
        """
        try:
            # Parse URL to get the path
            from urllib.parse import urlparse, unquote

            parsed = urlparse(url)
            path = unquote(parsed.path)

            # Extract extension
            extension = Path(path).suffix.lower()

            # Default to .mp3 if no extension found
            if not extension or extension not in [
                f".{fmt}" for fmt in self.supported_formats
            ]:
                extension = ".mp3"

            return extension
        except Exception:
            return ".mp3"

    def convert_to_mp3(self, input_path: str) -> str:
        """Convertit un fichier audio en MP3 standard"""
        try:
            import subprocess
            output_path = input_path + ".converted.mp3"
            
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-acodec', 'libmp3lame',
                '-ac', '1',  # Mono
                '-ar', '16000',  # 16kHz
                '-q:a', '2',  # Qualité moyenne
                output_path
            ], check=True)
            
            return output_path
        except Exception as e:
            raise AudioFormatError(f"Conversion en MP3 échouée: {str(e)}")
