"""Module for setting up vectorstores, retriever, reranker and a function for the chain."""

import asyncio
import json
import re
from threading import Thread
import re
import numpy as np
import tiktoken
import torch
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,  # type: ignore
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from transformers import AutoProcessor, AutoModelForImageTextToText

from src.utils import compute_similarity, delete_unused_sources
import concurrent.futures

class Retriever:
    def __init__(self, config, prompt: str, data_source: str):
        self.config = config
        self.embedding_model = EmbeddingsRedundantFilter("sentence-transformers/all-MiniLM-L6-v2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/gemma-3n-e2b-it"
        self.processor, self.model = self._initialize_model()
        self.llm = Ollama(
            model="gemma3:4b",
            temperature=0.4
        )
        self.prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["query"],
            partial_variables={"data_source": data_source},
        )
        self.pipeline = self.prompt_template | self.llm

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

    def get_llm_output(self, query: str) -> str:
        result = self.pipeline.invoke({"query": query})
        return result.content if hasattr(result, "content") else str(result)

    def parse_llm_output(self, llm_output: str) -> dict:
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not match:
            raise ValueError("JSON introuvable dans la sortie")
        data = json.loads(match.group())
        return {k: data.get(k, "") for k in ["query", "status", "nature_juridique", "legal_field", "article_num"]}

    def get_legal_fields(self, source: str):
        legal_fields = []
        if source == "labor_law":
            legal_fields.append("droit du travail")
        elif source == "national_transport":
            legal_fields.append("droit du transport")
        elif source == "insurance":
            legal_fields.append("droit des assurances")
        elif source == "uemoa":
            legal_fields.append("réglementation uemoa")
        elif source == "ohada":
            legal_fields.append("réglementation ohada")
        elif source == "digital_legislation":
            legal_fields.append("droit numérique")
        elif source == "jurisprudence":
            legal_fields.append("jurisprudence")
        return {"legal_field": ", ".join(legal_fields)}

    def orchestrator(self, llm_output: str, source: str) -> dict:
        parsed_output = self.parse_llm_output(llm_output)
        legal_fields = self.get_legal_fields(source)["legal_field"]
        parsed_output["legal_field"] = legal_fields
        if source == "jurisprudence":
            parsed_output["nature_juridique"] = ""
            parsed_output["status"] = ""
            parsed_output["article_num"] = ""
        return parsed_output

    def build_milvus_expr(self, filters: dict) -> str:
        champs = ["status", "nature_juridique", "legal_field", "article_num"]
        clauses = []
        for cle in champs:
            valeur = filters.get(cle, "")
            if isinstance(valeur, str):
                valeurs = [v.strip() for v in valeur.split(",") if v.strip()]
            elif isinstance(valeur, (list, tuple, set)):
                valeurs = [str(v).strip() for v in valeur if str(v).strip()]
            else:
                continue
            if not valeurs:
                continue
            if len(valeurs) == 1:
                clauses.append(f'{cle} == "{valeurs[0]}"')
            else:
                sous = " or ".join(f'{cle} == \"{v}\"' for v in valeurs)
                clauses.append(f"({sous})")
        return " and ".join(clauses)
    # Function to create retrievers dynamically based on the selected sources
    # Function to create retrievers dynamically based on the selected sources
    def create_vectorstore(self, source):
        if source not in self.config.COLLECTION_NAMES:
            raise ValueError(
                f"Source '{source}' non reconnue. Choisissez parmi : {list(self.config.COLLECTION_NAMES.keys())}"
            )
        return Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.config.COLLECTION_NAMES[source],
            connection_args=self.config.CONNECTION_ARGS,
            index_params=self.config.INDEX_PARAMS,
            search_params=self.config.SEARCH_PARAMS,
        )
        
    def create_retriever_for_source(self, source, top_k: int = 100, similarity_threshold: float = 0.4, expr: str = None):
        vectorstore = self.create_vectorstore(source)
        compressor = DocumentCompressorPipeline(
             transformers=[EmbeddingsRedundantFilter(embeddings=self.embedding_model, similarity_threshold=1)])
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,         
                "similarity_threshold": similarity_threshold,
                "param": self.config.SEARCH_PARAMS,
                "expr": expr,
            },
        )
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    
    def retrieve_docs_from_source(self, query: str, source: str, top_k: int = 100, similarity_threshold: float = 0.4, expr: str = None):
        # llm_output = self.get_llm_output(query)
        # filters = self.orchestrator(llm_output, source)
        # expr = self.build_milvus_expr(filters)
        retriever = self.create_retriever_for_source(
            source=source,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            expr=expr 
        )
        try:
            return retriever.invoke(query)
        except Exception as e:
            error_msg = str(e)
            if "field" in error_msg and "not exist" in error_msg:
                print("Champ inexistant détecté, relance sans expr.")
                retriever = self.create_retriever_for_source(
                    source=source,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    expr=""
                )
                return retriever.invoke(query)
            raise

    def format_docs(self, docs, max_context_size=100_000, separator="\n\n"):
        encoder = tiktoken.get_encoding("cl100k_base")
        context = ""
        for idx, doc in enumerate(docs):
            nature = doc.metadata['nature_juridique'].lower()
            if nature == "jurisprudence":
                if len(encoder.encode(context)) < max_context_size:
                    context += (
                        f"[{idx + 1}]. {doc.page_content.strip()}\n"
                        + f"Number: {idx + 1}\n"
                        + f"Numero de registre: {doc.metadata['text_juridique_num']}\n"
                        + f"Juridiction: {doc.metadata['juridiction']}\n"
                        + f"Type du texte juridique: {doc.metadata['nature_juridique']}\n"
                        + f"Date: {doc.metadata['date']}\n"
                        + f"Text juridique: {doc.metadata['text_juridique_name']}\n"
                    )
            else:
                if len(encoder.encode(context)) < max_context_size:
                    context += (
                        f"[{idx + 1}]. {doc.page_content.strip()}\n"
                        + f"Number: {idx + 1}\n"
                        + f"Article: {doc.metadata['article_num']}\n"
                        + f"Status: {doc.metadata['status']}\n"
                        + f"Type du texte juridique: {doc.metadata['nature_juridique']}\n"
                        + f"Texte juridique: {doc.metadata['text_juridique_name']}\n"
                        + separator
                    )
        if not context and docs:
            doc = docs[0]
            nature = doc.metadata['nature_juridique'].lower()
            if nature == "jurisprudence":
                context += (
                    f"[1]. {doc.page_content.strip()}\n"
                    + "Number: 1\n"
                    + f"Numero de registre: {doc.metadata['text_juridique_num']}\n"
                    + f"Juridiction: {doc.metadata['juridiction']}\n"
                    + f"Type du texte juridique: {doc.metadata['nature_juridique']}\n"
                    + f"Date: {doc.metadata['date']}\n"
                    + f"Text juridique: {doc.metadata['text_juridique_name']}\n"
                )
            else:
                context = (
                    f"[1]. {doc.page_content}\n"
                    + "Number: 1\n"
                    + f"Article: {doc.metadata['article_num']}\n"
                    + f"Status: {doc.metadata['status']}\n"
                    + f"Type du texte juridique: {doc.metadata['nature_juridique']}\n"
                    + f"Texte juridique: {doc.metadata['text_juridique_name']}\n"
                    + separator
                )
        return context.strip()
    

    # def rerank_docs(
    #     self,
    #     query: str,
    #     docs: list[Document],
    # ):
    #     """Re-rank documents according its relevances to the user query and return the top k."""
    #     query_emb = self.embedding_model.embed_query(query)
    #     similarities = np.asarray(
    #         list(
    #             (
    #                 map(
    #                     lambda doc: compute_similarity(
    #                         query_emb, doc.state["embedded_doc"]  # type: ignore
    #                     ),
    #                     docs,
    #                 )
    #             )
    #         )
    #     )
    #     return [docs[i] for i in similarities.argsort()[::-1]]

    def rerank_docs_(
            self,
            query: str,
            docs: list[Document],
        ):
            """Re-rank documents using parallel processing for speed."""
            query_emb = self.embedding_model.embed_query(query)
            def calculate_similarity(doc):
                return compute_similarity(query_emb, doc.state["embedded_doc"])
            with concurrent.futures.ThreadPoolExecutor() as executor:
                similarities_iterator = executor.map(calculate_similarity, docs)
                similarities = np.asarray(list(similarities_iterator))
            sorted_indices = similarities.argsort()[::-1]
            for i in sorted_indices[:200]:
                doc = docs[i]
                similarity_score = similarities[i]
                legal_field = doc.metadata.get("legal_field", "Non disponible")
                print(f"  Score de similarité: {similarity_score:.4f}, Domaine juridique (legal_field): {legal_field}")
            print("----------------------------------------------------------------------------------\n")
            return [docs[i] for i in sorted_indices]

    def filter_and_prioritize_sources(self, docs: list[Document], top_k: int) -> list[Document]:
            final_docs = []
            jurisprudence_count = 0
            max_jurisprudence = 7
            for doc in docs:
                is_jurisprudence = 'jurisprudence' in doc.metadata.get('legal_field', '').lower()
                if is_jurisprudence:
                    if jurisprudence_count < max_jurisprudence:
                        final_docs.append(doc)
                        jurisprudence_count += 1
                else:
                    final_docs.append(doc)
            return final_docs[:top_k]
    
    def _retrieve_for_single_source(self, query: str, source: str, llm_output: str, top_k: int, similarity_threshold: float) -> list[Document]:
        filters = self.orchestrator(llm_output, source)
        expr = self.build_milvus_expr(filters)
        return self.retrieve_docs_from_source(query, source, top_k, similarity_threshold, expr)
    
    async def _retrieve_for_single_source_async(self, query: str, source: str, llm_output: str, top_k: int, similarity_threshold: float) -> list[Document]:
        print("ok")
        return await asyncio.to_thread(
            self._retrieve_for_single_source,
            query,
            source,
            llm_output,
            top_k,
            similarity_threshold
        )

    async def source_retriever_chain_async(
        self, query: str, sources: list[str], top_k: int, similarity_threshold: float
    ) -> tuple[str, list[Document]]:
        llm_output = self.get_llm_output(query)
        tasks = []
        for source in sources:
            tasks.append(
                self._retrieve_for_single_source_async(query, source, llm_output, top_k, similarity_threshold)
            )
        list_of_docs_lists = await asyncio.gather(*tasks)
        docs = [doc for sublist in list_of_docs_lists for doc in sublist]
        docs = self.rerank_docs_(query=query, docs=docs)
        docs = self.filter_and_prioritize_sources(docs, top_k)
        context = self.format_docs(docs, max_context_size=100_000, separator="\n\n")
        return context, docs

    async def source_retriever_chain(
        self, query: str, sources: list[str], top_k, similarity_threshold
    ) -> tuple[str, list[Document]]:
        """Return all relevent documents to asnwer user query according the selected source"""
        # docs = []
        tasks = []
        llm_output = self.get_llm_output(query)
        for source in sources:
            filters = self.orchestrator(llm_output, source)
            expr = self.build_milvus_expr(filters)
            tasks.append(self.retrieve_docs_from_source(query, source, top_k, similarity_threshold, expr))
            # docs.append(self.retrieve_docs_from_source(query, source, top_k, similarity_threshold, expr))
        list_of_docs_lists = await asyncio.gather(*tasks)
        docs = [doc for sublist in list_of_docs_lists for doc in sublist]
        docs = self.rerank_docs_(query=query, docs=docs)
        docs = self.filter_and_prioritize_sources(docs, top_k)
        context = self.format_docs(docs, max_context_size=100_000, separator="\n\n")
        return context, docs


def create_llm(
    model,
    temperature,
    max_retries,
    callbacks: list = [],
) -> Ollama:
    return Ollama(  
        model=model,
        temperature=temperature,
        callbacks=callbacks,
    ).with_retry(
        wait_exponential_jitter=True,
        stop_after_attempt=max_retries,
    ) 


def answer_chain(
    query: str,
    sources: list[str],
    prompt: PromptTemplate,
    llm,
    retriever: Retriever,
    similarity_threshold: float = 0.4,
    top_k: int = 200,
) -> tuple[str, list[dict[str, str]]]:
    """Get the chain for the given retriever, prompt, and model."""

    context, docs = asyncio.run(retriever.source_retriever_chain_async(
    query, sources, top_k, similarity_threshold
    ))
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})
    answer = re.split(r"(-{3,})", answer)[0]
    documents = delete_unused_sources(answer, docs)
    
    return {
        "documents": documents,
        "answer": answer,
    }


def start_generation(chain, context, query):
    thread = Thread(
        target=chain.invoke, kwargs={"input": {"context": context, "query": query}}
    )
    thread.start()


END_GENERATION_TOKEN = "[END_GENERATION]"


async def stream_answer_chain(chain, context, query, docs, streamer_queue):
    start_generation(chain, context, query)

    # collect all token to parse used source at the end.
    collected_answer = ""
    while True:
        value = streamer_queue.get()
        
        # Check for the stop signal, which is None in our case
        if value == None:
            # If stop signal is found break the loop
            break
        collected_answer += value

        # Else yield the value
        yield value
        # statement to signal the queue that task is done
        streamer_queue.task_done()

        # guard to make sure we are not extracting anything from
        # empty queue

        await asyncio.sleep(0.03)
        
    documents = delete_unused_sources(collected_answer, docs)
    yield END_GENERATION_TOKEN + json.dumps({"source": documents})