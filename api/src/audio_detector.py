"""
Module AudioDetector pour la détection et validation d'URLs audio.

Ce module fournit des fonctionnalités pour détecter si une chaîne de requête contient
une URL audio et valider que l'URL est accessible et pointe vers un fichier audio valide.
"""

import re
import requests
from typing import Optional
from urllib.parse import urlparse


class AudioDetector:
    """
    Classe pour la détection et validation d'URLs audio.

    Cette classe fournit des méthodes statiques pour déterminer si une chaîne donnée
    contient une URL audio et pour valider que l'URL est accessible.
    """

    # Extensions de fichiers audio supportées
    AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "aac", "wma", "opus"}

    # Pattern regex pour détecter les URLs avec extensions audio
    AUDIO_URL_PATTERN = re.compile(
        r"https?://[^\s]+\.(?:"
        + "|".join(AUDIO_EXTENSIONS)
        + r")(?:\?[^\s]*)?(?:#[^\s]*)?$",
        re.IGNORECASE,
    )

    @staticmethod
    def is_audio_url(query: str) -> bool:
        """
        Détecte si la chaîne de requête contient une URL audio.

        Utilise des patterns regex pour identifier les URLs se terminant par des extensions
        de fichiers audio communes.

        Args:
            query (str): La chaîne de requête à analyser

        Returns:
            bool: True si la requête semble être une URL audio, False sinon
        """
        if not query or not isinstance(query, str):
            return False

        # Supprime les espaces et vérifie si cela correspond au pattern d'URL audio
        query = query.strip()
        return bool(AudioDetector.AUDIO_URL_PATTERN.match(query))

    @staticmethod
    def validate_audio_url(url: str, timeout: int = 10) -> bool:
        """
        Valide qu'une URL audio est accessible en utilisant une requête HTTP HEAD.

        Effectue une requête HEAD pour vérifier si l'URL est accessible et
        valide optionnellement le type de contenu si fourni par le serveur.

        Args:
            url (str): L'URL audio à valider
            timeout (int): Timeout de la requête en secondes (défaut: 10)

        Returns:
            bool: True si l'URL est accessible, False sinon
        """
        if not url or not isinstance(url, str):
            return False

        try:
            # Parse l'URL pour s'assurer qu'elle est valide
            parsed = urlparse(url.strip())
            if not parsed.scheme or not parsed.netloc:
                return False

            # Effectue une requête HEAD pour vérifier l'accessibilité
            response = requests.head(
                url.strip(),
                timeout=timeout,
                allow_redirects=True,
                headers={"User-Agent": "IrokoAPI-AudioProcessor/1.0"},
            )

            # Vérifie si la requête a réussi
            if response.status_code != 200:
                return False

            # Optionnel: Vérifie le type de contenu si fourni
            content_type = response.headers.get("content-type", "").lower()
            if content_type:
                # Types MIME audio courants
                audio_mime_types = {"audio/", "application/ogg", "video/ogg"}
                if not any(mime in content_type for mime in audio_mime_types):
                    # Si le type de contenu est fourni mais n'est pas lié à l'audio,
                    # retourne quand même True car certains serveurs peuvent ne pas définir les bons types MIME
                    pass

            return True

        except (requests.RequestException, ValueError, Exception):
            # Toute erreur réseau, timeout ou erreur de parsing signifie que l'URL n'est pas valide
            return False

    @staticmethod
    def extract_audio_url(query: str) -> Optional[str]:
        """
        Extrait l'URL audio d'une chaîne de requête si présente.

        Args:
            query (str): La chaîne de requête à analyser

        Returns:
            Optional[str]: L'URL audio extraite si trouvée, None sinon
        """
        if not AudioDetector.is_audio_url(query):
            return None

        return query.strip()
