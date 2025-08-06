"""Module for defining the utility functions."""

import re

from scipy.spatial import distance


def compute_similarity(vect1, vect2):
    return 1 - distance.cosine(vect1, vect2)


def get_answer_citations(answer: str) -> list[int]:
    citations = re.findall(r"\[\(?\d+\)?\]", answer)
    source_used_for_answer = [
        int(num) for citation in citations for num in re.findall(r"\d+", citation)
    ]
    return sorted(list(set(source_used_for_answer)))


def delete_unused_sources(answer, docs) -> list[dict[str, str]]:
    effective = get_answer_citations(answer)
    documents = []

    for i in effective:
        if 1 <= i <= len(docs):  # Vérifie si l'index est dans la plage valide
            doc = docs[i - 1]
            doc = doc.metadata
            doc.update({"id": str(i)})
            documents.append(doc)

    return documents
