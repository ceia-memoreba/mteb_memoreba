from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class RRIPClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RRIP",
        description="Rhetorical annotations from judicial sentences from the Court of Justice of Mato Grosso do Sul (Brazil). Rhetorical role identification (RRI) is an NLP task that consists of labeling the sentences of a document according to a given set of semantic functions (rhetorical roles).",
        reference="https://link.springer.com/chapter/10.1007/978-3-030-91699-2_38",
        dataset={
            "path": "eduagarcia/PortuLex_benchmark",
            "name": "rrip",
            "revision": "cddad88112fb6d0bb4f66cb58a362f08b85033e1",
        },
        type="Classification",
        category="s2s",
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="f1",
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{aragy_rhetorical_2021,
    series = {Lecture {Notes} in {Computer} {Science}},
    title = {Rhetorical {Role} {Identification} for {Portuguese} {Legal} {Documents}},
    isbn = {978-3-030-91699-2},
    doi = {10.1007/978-3-030-91699-2_38},
    booktitle = {Intelligent {Systems}},
    publisher = {Springer International Publishing},
    author = {Aragy, Roberto and Fernandes, Eraldo Rezende and Caceres, Edson Norberto},
    editor = {Britto, André and Valdivia Delgado, Karina},
    year = {2021},
    pages = {557--571},
}
""",
        n_samples={"train": 8257, "validation": 1053, "test": 1474},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
