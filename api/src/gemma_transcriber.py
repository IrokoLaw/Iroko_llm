import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText
import requests
import tempfile
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GemmaAudioTranscriber:
    def __init__(self, config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/gemma-3n-e2b-it"  # Version correcte
        self.processor, self.model = self._initialize_model()
        
    def _initialize_model(self):
        """Initialisation avec gestion mémoire optimisée"""
        logger.info(f"Loading {self.model_name} on {self.device}")
        
        # Configuration pour gérer la mémoire limitée
        processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Options pour systèmes avec RAM limitée
        if self.device == "cpu":
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,  # Pas de device_map automatique
                low_cpu_mem_usage=True,  # Optimisation mémoire CPU
                load_in_8bit=False,  # Pas de quantification sur CPU
            )
        else:
            # Pour GPU avec mémoire limitée
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="balanced_low_0",  # Distribution équilibrée
                load_in_8bit=True,  # Quantification 8-bit pour économiser
            )
        
        return processor, model


    def _prepare_audio(self, audio_path: str) -> str:
        """Prépare l'audio selon les specs Gemma 3n"""
        # Gemma 3n veut: mono, 16kHz, float32 dans [-1,1]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Conversion mono si stéréo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resampling à 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            
        # Normalisation à [-1,1]
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Sauvegarde temporaire au bon format
        temp_path = tempfile.mktemp(suffix='.wav')
        torchaudio.save(temp_path, waveform, 16000)
        
        return temp_path

    # def transcribe_audio(self, audio_url: str) -> str:
    #     """Transcription avec la méthode correcte de Gemma 3n"""
    #     temp_files = []
        
    #     try:
    #         # 1. Téléchargement avec gestion des redirections
    #         logger.info(f"Téléchargement depuis {audio_url}")
            
    #         response = requests.get(audio_url, timeout=30, allow_redirects=True)
    #         response.raise_for_status()
            
    #         # 2. Sauvegarde temporaire
    #         temp_audio = tempfile.mktemp(suffix='.mp3')
    #         temp_files.append(temp_audio)
            
    #         with open(temp_audio, 'wb') as f:
    #             f.write(response.content)
                
    #         logger.info(f"Audio téléchargé: {len(response.content)} bytes")
            
    #         # 3. Préparation audio selon specs Gemma
    #         prepared_audio = self._prepare_audio(temp_audio)
    #         temp_files.append(prepared_audio)
            
    #         # 4. Construction du message pour Gemma 3n
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "audio", "audio": prepared_audio},
    #                     {"type": "text", "text": "Transcris cet audio en français avec précision."}
    #                 ]
    #             }
    #         ]
            
    #         # 5. Traitement avec le processor
    #         inputs = self.processor.apply_chat_template(
    #             messages,
    #             add_generation_prompt=True,
    #             tokenize=True,
    #             return_dict=True,
    #             return_tensors="pt"
    #         )
            
    #         inputs = inputs.to(self.device)
            
    #         # 6. Génération
    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 **inputs, 
    #                 max_new_tokens=512,
    #                 temperature=0.1,
    #                 do_sample=False
    #             )
            
    #         # 7. Décodage
    #         transcription = self.processor.batch_decode(
    #             outputs, 
    #             skip_special_tokens=True,
    #             clean_up_tokenization_spaces=True
    #         )[0]
            
    #         return transcription
            
    #     except Exception as e:
    #         logger.error(f"Erreur transcription: {str(e)}")
    #         raise
            
    #     finally:
    #         # Nettoyage des fichiers temporaires
    #         for temp_file in temp_files:
    #             if os.path.exists(temp_file):
    #                 os.remove(temp_file)


    def transcribe_audio(self, audio_path: str) -> str:
        """Transcription avec la méthode correcte de Gemma 3n"""
        temp_files = []
        
        try:
            # Si c'est une URL, on télécharge d'abord
            if audio_path.startswith(('http://', 'https://')):
                logger.info(f"Téléchargement depuis {audio_path}")
                response = requests.get(audio_path, timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                # Sauvegarde temporaire
                temp_audio = tempfile.mktemp(suffix='.mp3')
                temp_files.append(temp_audio)
                
                with open(temp_audio, 'wb') as f:
                    f.write(response.content)
                audio_to_process = temp_audio
            else:
                # Si c'est déjà un chemin de fichier local
                audio_to_process = audio_path
                    
            logger.info(f"Traitement du fichier audio: {audio_to_process}")
            
            # Préparation audio selon specs Gemma
            prepared_audio = self._prepare_audio(audio_to_process)
            temp_files.append(prepared_audio)
            
            # Reste du code inchangé...
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": prepared_audio},
                        {"type": "text", "text": "Transcris cet audio en français avec précision."}
                    ]
                }
            ]
            
            # Traitement avec le processor
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            inputs = inputs.to(self.device)
            
            # Génération
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Décodage
            transcription = self.processor.batch_decode(
                outputs, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Erreur transcription: {str(e)}")
            raise
            
        finally:
            # Nettoyage des fichiers temporaires (sauf le fichier d'origine si c'était un chemin local)
            for temp_file in temp_files:
                if os.path.exists(temp_file) and temp_file != audio_path:
                    os.remove(temp_file)
