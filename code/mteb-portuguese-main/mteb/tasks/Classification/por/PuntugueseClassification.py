from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PuntugueseClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Puntuguese",
        description="Puntuguese is a corpus of punning texts in Portuguese, including jokes in Brazilian and European Portuguese. The data has been manually gathered and curate according to our guidelines.",
        reference="https://github.com/Superar/Puntuguese",
        dataset={
            "path": "Superar/Puntuguese",
            "revision": "47ab567cd033eafdcfcc8520c953d879c2e2bd91",
        },
        type="Classification",
        category="s2s",
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="accuracy",
        form=["written"],
        domains=["Web", "Social"],
        task_subtypes=[],
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR", "pt-PT"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{InacioEtAl2024,
    title     = {Puntuguese: A Corpus of Puns in {{P}}ortuguese with Micro-editions},
    author    = {In{\'a}cio, Marcio Lima and {Wick-pedro}, Gabriela and Ramisch, Renata and Esp{\'i}rito Santo, Lu{\'i}s and Chacon, Xiomara S. Q. and Santos, Roney and Sousa, Rog{\'e}rio and Anchi{\^e}ta, Rafael and Gon{\c c}alo Oliveira, Hugo},
    year      = {2024},
    note      = {Accepted to LREC-COLING 2024}
}
""",
        n_samples={'train': 3990, 'validation': 570, 'test': 1140},
        avg_character_length={'train': 66.48, 'validation': 65.44, 'test': 66.47},
    )
