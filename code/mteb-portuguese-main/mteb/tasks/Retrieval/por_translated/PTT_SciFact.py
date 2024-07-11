from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_SciFact(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_SciFact",
        dataset={
            "path": "pt-mteb/translated_scifact",
            "revision": "main",
        },
        description="PTT_SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        type="Retrieval",
        category="s2p",
        eval_splits=["train", "test"],
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
