"""Main module for configuring and running the IrokoAPI application."""

from enum import Enum
from queue import Queue
from typing import Annotated, List

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
import asyncio

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
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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
            description="A string as a question to ask to the agent.",
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
    """Root that take user query for chatting with the Iroko.

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
    llm = ChatOpenAI(
        model=model.value, temperature=temperature, openai_api_key=config.OPENAI_API_KEY
    )
    retriever = Retriever(config=config, prompt=prompt, data_source=data_source)
    source_list = [s.value for s in sources]

    source_list = (
        [source.value for source in Sources if source.value != "all"]
        if "all" in source_list
        else (source_list if "jurisprudence" in source_list else source_list + ["jurisprudence"])
    )
    answer, documents = answer_chain(
        query,
        source_list,
        qa_prompt,
        llm,
        retriever,
        similarity_threshold,
        top_k,
    )

    return {
        "documents": documents,
        "answer": answer,
    }


@app.get("/stream_question_answering", tags=["QA"])
def stream_question_answering(
    token: Annotated[str, Depends(oauth2_scheme)],
    query: Annotated[
        str,
        Query(
            title="The user question",
            description="A string as a question to ask to the agent.",
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
        else (source_list if "jurisprudence" in source_list else source_list + ["jurisprudence"])
    )
    # context, docs =  retriever.source_retriever_chain(
    #     query, source_list, top_k, similarity_threshold
    # )
    context, docs = asyncio.run(retriever.source_retriever_chain_async(
    query, source_list, top_k, similarity_threshold
    ))
    chain = qa_prompt | llm

    response = StreamingResponse(
        stream_answer_chain(chain, context, query, docs, streamer_queue),
        media_type="text/event-stream",
    )
    return response