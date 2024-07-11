from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_AILACasedocs(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_AILACasedocs",
        description="The task is to retrieve the case document that most closely matches or is most relevant to the scenario described in the provided query.",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "pt-mteb/translated_AILA_casedocs",
            "revision": "main",
        },
        type="Retrieval",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
