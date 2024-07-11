from __future__ import annotations

from datasets import load_dataset, Features, Value

from mteb.abstasks import AbsTaskRetrieval, TaskMetadata


class Pira2Retrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Pira2Retrieval",
        dataset={
            "path": "mteb-pt/pira2_retrieval",
            "revision": "c9e8283463ae0e429501ebbdcc773b1b0bcc1c6d",
        },
        description="Pirá is a crowdsourced reading comprehension dataset on the ocean, the Brazilian coast, and climate change.",
        reference="https://dl.acm.org/doi/pdf/10.1145/3459637.3482012",
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2022-01-01"),  # approximate guess
        form=["written"],
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        socioeconomic_status=None,
        annotations_creators="human-annotated",
        dialect=None,
        text_creation="found",
        bibtex_citation="""
@inproceedings{10.1145/3459637.3482012,
    author = {Paschoal, Andr\'{e} F. A. and Pirozelli, Paulo and Freire, Valdinei and Delgado, Karina V. and Peres, Sarajane M. and Jos\'{e}, Marcos M. and Nakasato, Fl\'{a}vio and Oliveira, Andr\'{e} S. and Brand\~{a}o, Anarosa A. F. and Costa, Anna H. R. and Cozman, Fabio G.},
    title = {Pir\'{a}: A Bilingual Portuguese-English Dataset for Question-Answering about the Ocean},
    year = {2021},
    isbn = {9781450384469},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3459637.3482012},
    doi = {10.1145/3459637.3482012},
    booktitle = {Proceedings of the 30th ACM International Conference on Information & Knowledge Management},
    pages = {4544–4553},
    numpages = {10},
    location = {Virtual Event, Queensland, Australia},
    series = {CIKM '21}
}""",
        n_samples={'test': 702},
        avg_character_length={'test': 1745.8}
    )

