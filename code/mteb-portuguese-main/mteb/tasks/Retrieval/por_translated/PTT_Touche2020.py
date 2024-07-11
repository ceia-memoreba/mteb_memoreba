from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_Touche2020(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_Touche2020",
        description="Touché Task 1: Argument Retrieval for Controversial Questions",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "pt-mteb/translated_touche2020",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=None,
        domains=None,
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators=None,
        dialect=None,
        text_creation=None,
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
