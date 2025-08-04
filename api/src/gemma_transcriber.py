"""
Classe GemmaTranscriber corrigée pour la transcription audio utilisant Gemma 3n-E2B.
Basée sur la documentation officielle Google.
"""

import os
import re
import logging
import torch
import torchaudio
import numpy as np
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import AudioConfig

logger = logging.getLogger(__name__)


class GemmaTranscriber:
    """
    Classe de transcription audio utilisant le modèle Gemma 3n-E2B.
    Version corrigée basée sur l'API officielle Google.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialise le GemmaTranscriber avec la configuration.

        Args:
            config: Instance AudioConfig, utilise la configuration par défaut si None
        """
        self.config = config or AudioConfig()
        self.device = self._detect_device()
        self.model = None
        self.processor = None
        self._initialize_model()

    def _detect_device(self) -> str:
        """
        Détecte le périphérique disponible (GPU/CPU) pour le traitement.

        Returns:
            Chaîne du périphérique ('cuda' ou 'cpu')
        """
        if self.config.use_gpu and torch.cuda.is_available():
            device = "cuda"
            logger.info(
                f"Using GPU for audio transcription: {torch.cuda.get_device_name()}"
            )
        else:
            device = "cpu"
            if self.config.use_gpu:
                logger.warning("GPU requested but not available, falling back to CPU")
            else:
                logger.info("Using CPU for audio transcription")

        return device

    def _initialize_model(self) -> None:
        """
        Initialize the Gemma 3n-E2B model and processor for audio transcription.
        Utilise l'API officielle recommandée par Google.
        """
        try:
            logger.info(
                f"Initializing Gemma 3n-E2B model: {self.config.gemma_model_id}"
            )

            # Utiliser AutoModelForImageTextToText comme recommandé par Google
            # pour les modèles multimodaux Gemma 3n
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.gemma_model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )

            if self.device == "cpu":
                self.model = self.model.to("cpu")

            # Set model to evaluation mode
            self.model.eval()

            # Load processor - celui-ci devrait fonctionner avec Gemma 3n
            self.processor = AutoProcessor.from_pretrained(
                self.config.gemma_model_id, 
                trust_remote_code=True
            )

            logger.info("Gemma 3n-E2B model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gemma model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using Gemma 3n-E2B model.
        Utilise l'approche officielle recommandée par Google.

        Args:
            audio_path: Path to the audio file to transcribe

        Returns:
            Transcribed text as string

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Starting transcription of audio file: {audio_path}")

            # Vérifier que le processor est disponible
            if self.processor is None:
                raise RuntimeError(
                    "Processor not initialized - cannot perform transcription"
                )

            # Load and preprocess audio
            audio_array, sampling_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio_array.shape[0] > 1:
                audio_array = torch.mean(audio_array, dim=0, keepdim=True)
            
            # Convert to numpy for processor
            audio_array = audio_array.squeeze().numpy()

            # Create the prompt for transcription (en français)
            prompt = "Transcris cet audio en français:"

            # Process inputs using the official approach
            inputs = self.processor(
                text=prompt,
                audio=audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )

            # Move inputs to device
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode the generated text
            # Skip the input tokens to get only the generated response
            input_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            generated_tokens = outputs[0][input_length:]
            raw_text = self.processor.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Clean and format the transcribed text
            cleaned_text = self.cleanup_and_format_text(raw_text)

            if not cleaned_text or not cleaned_text.strip():
                raise RuntimeError("Transcription produced empty text")

            logger.info(
                f"Transcription completed successfully. Text length: {len(cleaned_text)} characters"
            )
            return cleaned_text

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {str(e)}")
            raise RuntimeError(f"Audio transcription failed: {str(e)}")

    def cleanup_and_format_text(self, raw_text: str) -> str:
        """
        Clean and format the raw transcribed text.

        Args:
            raw_text: Raw text from transcription

        Returns:
            Cleaned and formatted text
        """
        if not raw_text or not raw_text.strip():
            return ""

        # Remove leading/trailing whitespace
        text = raw_text.strip()

        # Remove common transcription artifacts
        text = re.sub(r"\[.*?\]", "", text)  # Remove bracketed content
        text = re.sub(r"\(.*?\)", "", text)  # Remove parenthetical content
        text = re.sub(r"<.*?>", "", text)  # Remove angle bracketed content

        # Remove multiple consecutive spaces (after artifact removal)
        text = re.sub(r"\s+", " ", text)

        # Fix common French punctuation issues
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # Remove space before punctuation
        text = re.sub(r"([,.!?;:])\s*", r"\1 ", text)  # Ensure space after punctuation

        # Capitalize first letter of sentences
        sentences = re.split(r"([.!?]+\s*)", text)
        formatted_sentences = []

        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Actual sentence content
                sentence = sentence.strip()
                if sentence:
                    sentence = (
                        sentence[0].upper() + sentence[1:]
                        if len(sentence) > 1
                        else sentence.upper()
                    )
                formatted_sentences.append(sentence)
            elif sentence.strip():  # Punctuation with potential spaces
                formatted_sentences.append(sentence.rstrip() + " ")

        text = "".join(formatted_sentences).strip()

        # Final cleanup
        text = text.strip()

        # Ensure text ends with proper punctuation
        if text and not text[-1] in ".!?":
            text += "."

        logger.debug(
            f"Text cleanup completed. Original length: {len(raw_text)}, Final length: {len(text)}"
        )
        return text

    def __del__(self):
        """
        Cleanup resources when object is destroyed.
        """
        if hasattr(self, "model") and self.model is not None:
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()

        # Clean up model and processor references
        self.model = None
        self.processor = None