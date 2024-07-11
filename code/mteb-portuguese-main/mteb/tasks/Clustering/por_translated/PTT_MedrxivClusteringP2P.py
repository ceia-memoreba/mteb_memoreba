from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class PTT_MedrxivClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PTT_MedrxivClusteringP2P",
        description="Clustering of titles+abstract from medrxiv. Clustering of 10 sets, based on the main category.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "pt-mteb/translated_medrxiv-clustering-p2p",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
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
        n_samples={"test": 375000},
        avg_character_length={"test": 1981.2},
    )
