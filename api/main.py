"""Main module for configuring and running the IrokoAPI application."""

from enum import Enum
from queue import Queue
from typing import Annotated, List
import logging

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from langchain_openai.chat_models import ChatOpenAI

import config
from fields import Sources, source_list
from src.chain import Retriever, answer_chain, create_llm, stream_answer_chain
from src.handlers import StreamingHandler
from src.prompts import qa_prompt, prompt, data_source
from src.audio_query_processor import (
    AudioQueryProcessor,
    AudioProcessingError,
    AudioURLError,
    AudioTranscriptionError,
)
import asyncio

# Configure logging
logger = logging.getLogger(__name__)


def get_local_model_path(model_name: str) -> str:
    """
    Retourne le chemin local du modèle s'il existe, sinon retourne le nom original.

    Args:
        model_name: Nom du modèle (ex: 'google/gemma-3n-e2b')

    Returns:
        Chemin local du modèle ou nom original
    """
    project_models_dir = os.path.join(os.path.dirname(__file__), ".models")

    # Convertir le nom du modèle au format de cache HF
    cache_name = model_name.replace("/", "--")
    model_cache_dir = os.path.join(project_models_dir, f"models--{cache_name}")

    if os.path.exists(model_cache_dir):
        # Chercher le snapshot le plus récent
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                # Prendre le premier snapshot disponible
                snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                logger.info(f"Using local model: {snapshot_path}")
                return snapshot_path

    logger.info(f"Local model not found, using online: {model_name}")
    return model_name


class Models(Enum):
    """Models available for IrokoAPI."""

    CHATGPT = "chatgpt-4o-latest"
    GPT4o = "gpt-4o-2024-11-20"
    GPT4o_mini = "gpt-4o-mini-2024-07-18"


app = FastAPI(
    title="IrokoAPI",
    description="""**Iroko** est un assistant juridique dédié aux professionnels du droit en Afrique. Il vise à leur offrir un accès facile et rapide à des informations juridiques exhaustives, pertinentes et à jour à travers une plateforme unique, simplifiant ainsi leur processus de recherche et d'analyse juridique.""",  # pylint: disable=line-too-long
    version="0.1.0",
    contact={
        "name": "Hans Ariel",
        "email": "hansearieldo@gmail.com",
    },
)
# Configure le cache Hugging Face local pour utiliser les modèles du projet
import os

project_models_dir = os.path.join(os.path.dirname(__file__), ".models")
if os.path.exists(project_models_dir):
    # Configurer les variables d'environnement pour Hugging Face
    os.environ["HF_HOME"] = project_models_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(project_models_dir, "hub")
    os.environ["HF_HUB_CACHE"] = os.path.join(project_models_dir, "hub")
    # Désactiver la vérification en ligne pour utiliser le cache local
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info(f"Using local models directory: {project_models_dir}")
else:
    logger.warning("Local models directory not found, using default HF cache")

# Initialise le processeur audio globalement pour éviter de le recréer à chaque requête
try:
    audio_config = config.AudioConfig()

    # Utiliser le modèle local s'il existe
    local_model_path = get_local_model_path(audio_config.gemma_model_id)
    if local_model_path != audio_config.gemma_model_id:
        # Créer une nouvelle config avec le chemin local
        audio_config.gemma_model_id = local_model_path
        logger.info(f"Using local model: {local_model_path}")
        # Désactiver le mode offline pour utiliser le chemin direct
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    audio_processor = AudioQueryProcessor(config=audio_config)
    logger.info("AudioQueryProcessor initialized successfully")
except Exception as e:
    logger.warning(f"AudioQueryProcessor initialization failed: {e}")
    audio_processor = None
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def preprocess_query(query: str) -> str:
    """
    Prétraite la requête pour gérer les URLs audio et les convertir en texte.

    Args:
        query: La requête d'entrée (texte ou URL audio)

    Returns:
        Requête texte traitée

    Raises:
        HTTPException: Si le traitement audio échoue
    """
    if not audio_processor:
        # Si le processeur audio n'est pas disponible, retourne la requête telle quelle
        logger.warning("Audio processor not available, treating query as text")
        return query

    try:
        # Traite la requête (retourne telle quelle si texte, ou transcrit si URL audio)
        processed_query = audio_processor.process_query(query)

        # Log si le traitement audio a eu lieu
        if audio_processor.is_audio_query(query):
            logger.info(
                f"Audio URL detected and transcribed: {len(processed_query)} characters"
            )

        return processed_query

    except AudioURLError as e:
        logger.error(f"Invalid audio URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_audio_url",
                "message": str(e),
                "supported_formats": (
                    audio_processor.get_supported_formats() if audio_processor else []
                ),
                "max_size_mb": (
                    audio_processor.get_max_file_size_mb() if audio_processor else 50
                ),
            },
        )
    except AudioTranscriptionError as e:
        logger.error(f"Audio transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "transcription_failed",
                "message": str(e),
                "suggestion": "Vérifiez la qualité de l'enregistrement audio et réessayez.",
            },
        )
    except AudioProcessingError as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": e.error_type, "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Unexpected error during query preprocessing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "preprocessing_failed",
                "message": "Erreur inattendue lors du traitement de la requête.",
            },
        )


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    if (form_data.password == config.IROKO_API_ACCESS_TOKEN) or (
        form_data.client_secret == config.IROKO_API_ACCESS_TOKEN
    ):
        return {"access_token": config.IROKO_API_ACCESS_TOKEN, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/source_list", tags=["SOURCES"])
def get_sources(token: Annotated[str, Depends(oauth2_scheme)]):
    """Returns the list of sources available for IrokoAPI."""
    if token != config.IROKO_API_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect api key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return source_list


@app.get("/question_answering", tags=["QA"])
def question_answering(
    token: Annotated[str, Depends(oauth2_scheme)],
    query: Annotated[
        str,
        Query(
            title="The user question",
            description="Une chaîne comme question à poser à l'agent, ou une URL audio (mp3, wav, m4a, ogg).",
        ),
    ],
    sources: Annotated[
        List[Sources],
        Query(
            title="The list of source.",
            description="The sources from which the answer is generated.",
        ),
    ] = [Sources.ALL],
    model: Annotated[
        Models,
        Query(
            title="Select OpenAI model",
            description="Select OpenAI model you want to use.",
        ),
    ] = Models.CHATGPT,
    temperature: Annotated[
        float,
        Query(
            title="The temperature",
            description="The temperature.",
        ),
    ] = 0.4,
    similarity_threshold: Annotated[
        float,
        Query(
            title="The similarity score threshold",
            description="The similarity score threshold.",
        ),
    ] = 0.4,
    top_k: Annotated[
        int,
        Query(
            title="The amount of results to return at retrieval step",
            description="The amount of results to return at retrieval step.",
        ),
    ] = 100,
):
    """Root that take user query for chatting with the IROKO.

    Args:<br>
        query (str): A string as a question to ask to the agent.<br>
        source (list[str]): The source from which the answer is generated. Defaults to law.<br>
        model (str): Select OpenAI model you want to use. Defaults to gpt-4o.<br>
        max_context_size (int, optional): The maximum number of tokens to be passed as context to LLM. Defaults to 3000.<br>
        temperature (float, optional): The temperature. Defaults to 0.7.<br>
        similarity_threshold (float, optional): The similarity score threshold. Defaults to 0.4.<br>
        top_k (int, optional): The amount of results to return at retrieval step. Defaults to 10.<br>

    Returns:<br>
        A dictionary contain llm generation and context source.
    """
    if token != config.IROKO_API_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect api key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Prétraite la requête pour gérer les URLs audio
    original_query = query
    processed_query = preprocess_query(query)
    is_audio_input = audio_processor and audio_processor.is_audio_query(original_query)

    llm = ChatOpenAI(
        model=model.value, temperature=temperature, openai_api_key=config.OPENAI_API_KEY
    )
    retriever = Retriever(config=config, prompt=prompt, data_source=data_source)
    source_list = [s.value for s in sources]

    source_list = (
        [source.value for source in Sources if source.value != "all"]
        if "all" in source_list
        else (
            source_list
            if "jurisprudence" in source_list
            else source_list + ["jurisprudence"]
        )
    )

    # Utilise la requête traitée pour la chaîne de réponse
    answer, documents = answer_chain(
        processed_query,
        source_list,
        qa_prompt,
        llm,
        retriever,
        similarity_threshold,
        top_k,
    )

    response = {
        "documents": documents,
        "answer": answer,
    }

    # Ajoute les métadonnées de traitement audio si applicable
    if is_audio_input:
        response["audio_metadata"] = {
            "original_query": original_query,
            "transcribed_text": processed_query,
            "transcription_length": len(processed_query),
            "audio_processed": True,
        }

    return response


@app.get("/stream_question_answering", tags=["QA"])
def stream_question_answering(
    token: Annotated[str, Depends(oauth2_scheme)],
    query: Annotated[
        str,
        Query(
            title="The user question",
            description="Une chaîne comme question à poser à l'agent, ou une URL audio (mp3, wav, m4a, ogg).",
        ),
    ],
    sources: Annotated[
        List[Sources],
        Query(
            title="The list of source.",
            description="The sources from which the answer is generated.",
        ),
    ] = [Sources.ALL],
    model: Annotated[
        Models,
        Query(
            title="Select OpenAI model",
            description="Select OpenAI model you want to use.",
        ),
    ] = Models.CHATGPT,
    temperature: Annotated[
        float,
        Query(
            title="The temperature",
            description="The temperature.",
        ),
    ] = 0.4,
    similarity_threshold: Annotated[
        float,
        Query(
            title="The similarity score threshold",
            description="The similarity score threshold.",
        ),
    ] = 0.3,
    top_k: Annotated[
        int,
        Query(
            title="The amount of results to return at retrieval step",
            description="The amount of results to return at retrieval step.",
        ),
    ] = 100,
):
    """Root that take user query for chatting with the Iroko API.

    Args:<br>
        query (str): A string as a question to ask to the agent.<br>
        source (list[str]): The source from which the answer is generated. Defaults to law.<br>
        model (str): Select OpenAI model you want to use. Defaults to gpt-4o.<br>
        max_context_size (int, optional): The maximum number of tokens to be passed as context to LLM. Defaults to 3000.<br>
        temperature (float, optional): The temperature. Defaults to 0.7.<br>
        similarity_threshold (float, optional): The similarity score threshold. Defaults to 0.4.<br>
        top_k (int, optional): The amount of results to return at retrieval step. Defaults to 10.<br>

    Returns:<br>
        A dictionary contain llm generation and context source.
    """
    if token != config.IROKO_API_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect api key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Prétraite la requête pour gérer les URLs audio
    original_query = query
    processed_query = preprocess_query(query)
    is_audio_input = audio_processor and audio_processor.is_audio_query(original_query)

    # Creating a Streamer queue
    streamer_queue = Queue()
    # Creating an object of custom handler
    stream_handler = StreamingHandler(streamer_queue)
    llm = create_llm(
        api_key=config.OPENAI_API_KEY,
        model=model,
        temperature=temperature,
        max_retries=config.NUM_RETRY,
        streaming=True,
        callbacks=[stream_handler],
    )
    retriever = Retriever(config=config, prompt=prompt, data_source=data_source)
    source_list = [s.value for s in sources]
    source_list = (
        [source.value for source in Sources if source.value != "all"]
        if "all" in source_list
        else (
            source_list
            if "jurisprudence" in source_list
            else source_list + ["jurisprudence"]
        )
    )

    # Utilise la requête traitée pour la récupération et le streaming
    context, docs = asyncio.run(
        retriever.source_retriever_chain_async(
            processed_query, source_list, top_k, similarity_threshold
        )
    )
    chain = qa_prompt | llm

    response = StreamingResponse(
        stream_answer_chain(chain, context, processed_query, docs, streamer_queue),
        media_type="text/event-stream",
    )

    # Ajoute les métadonnées audio aux headers de réponse si applicable
    if is_audio_input:
        response.headers["X-Audio-Processed"] = "true"
        response.headers["X-Original-Query"] = (
            original_query[:100] + "..."
            if len(original_query) > 100
            else original_query
        )
        response.headers["X-Transcription-Length"] = str(len(processed_query))

    return response
