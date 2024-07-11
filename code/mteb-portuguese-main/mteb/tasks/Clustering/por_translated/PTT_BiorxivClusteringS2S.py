from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class PTT_BiorxivClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PTT_BiorxivClusteringS2S",
        description="Clustering of titles from biorxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "pt-mteb/translated_biorxiv-clustering-s2s",
            "revision": "main",
        },
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="v_measure",
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
        n_samples={"test": 75000},
        avg_character_length={"test": 101.6},
    )
