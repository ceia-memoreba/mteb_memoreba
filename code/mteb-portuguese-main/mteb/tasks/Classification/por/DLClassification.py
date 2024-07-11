from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class DLClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="datalawyer-frases-v0_4",
        description="",
        reference=None,
        dataset={
            "path": "datalawyer/datalawyer-frases-v0_4",
            "revision": "4c26a754af2c4cce07689042d4955ea2392270c1",
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
        bibtex_citation=None,
        n_samples={"train": 47836, "test": 10250},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
