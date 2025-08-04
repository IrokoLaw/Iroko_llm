# Fichiers de la Fonctionnalité Audio

## Fichiers Principaux

### 1. `src/audio_detector.py`

- **Rôle** : Détection et validation d'URLs audio
- **Classe principale** : `AudioDetector`
- **Fonctions clés** :
  - `is_audio_url()` : Détecte les URLs audio
  - `validate_audio_url()` : Vérifie l'accessibilité
  - `extract_audio_url()` : Extrait l'URL

### 2. `src/audio_downloader.py`

- **Rôle** : Téléchargement et validation de fichiers
- **Classe principale** : `AudioDownloader`
- **Fonctions clés** :
  - `download_audio()` : Télécharge le fichier
  - `validate_audio_format()` : Valide avec librosa
  - `cleanup_temp_file()` : Nettoie les fichiers temporaires

### 3. `src/gemma_transcriber.py`

- **Rôle** : Transcription avec Gemma 3n-E2B
- **Classe principale** : `GemmaTranscriber`
- **Fonctions clés** :
  - `transcribe_audio()` : Transcrit en français
  - `_preprocess_audio()` : Prétraite l'audio
  - `cleanup_and_format_text()` : Formate le texte

### 4. `src/audio_query_processor.py`

- **Rôle** : Orchestrateur principal
- **Classe principale** : `AudioQueryProcessor`
- **Fonctions clés** :
  - `process_query()` : Traite texte ou audio
  - Gestion d'erreurs complète
  - Nettoyage automatique des ressources

### 5. `main.py` (modifié)

- **Ajouts** : Intégration dans les routes API
- **Fonction ajoutée** : `preprocess_query()`
- **Routes modifiées** :
  - `/question_answering`
  - `/stream_question_answering`

## Configuration

### `config.py` (modifié)

- Ajout de la classe `AudioConfig`
- Variables d'environnement audio

### `.env` (à configurer)

```env
AUDIO_MAX_FILE_SIZE_MB=50
AUDIO_DOWNLOAD_TIMEOUT_SECONDS=30
AUDIO_GEMMA_MODEL_ID=google/gemma-3n-e2b
AUDIO_USE_GPU=true
AUDIO_TEMP_DIR=/tmp/iroko_audio
```
