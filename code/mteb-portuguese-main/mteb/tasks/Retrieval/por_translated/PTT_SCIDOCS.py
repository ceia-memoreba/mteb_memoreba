from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_SCIDOCS(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_SCIDOCS",
        dataset={
            "path": "pt-mteb/translated_scidocs",
            "revision": "main",
        },
        description=(
            "SciDocs, a new evaluation benchmark consisting of seven document-level tasks ranging from citation"
            " prediction, to document classification and recommendation."
        ),
        reference="https://allenai.org/data/scidocs",
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
