from enum import Enum


class Sources(str, Enum):
    """Sources available for IrokoAPI."""

    ALL = "all"
    INSURANCE = "insurance"
    UEMOA = "uemoa"
    OHADA = "ohada"
    JURISPRUDENCE = "jurisprudence"


source_list = [
    {
        "tag": "All",
        "items": [{"source": "all", "label": "Toutes les sources"}],
    },
    {
        "tag": "Assurances",
        "items": [
            {
                "source": "insurance",
                "label": "Droit des assurances",
            }
        ],
    },
    {
        "tag": "UEMOA",
        "items": [
            {
                "source": "uemoa",
                "label": "Réglementation UEMOA",
            }
        ],
    },
    {
        "tag": "OHADA",
        "items": [
            {
                "source": "ohada",
                "label": "Réglementation OHADA",
            }
        ],
    },
    {
            "tag": "Jurisprudence",
            "items": [
                {
                    "source": "jurisprudence",
                    "label": "jurisprudence Ivoirienne",
                }
            ],
    },
]
