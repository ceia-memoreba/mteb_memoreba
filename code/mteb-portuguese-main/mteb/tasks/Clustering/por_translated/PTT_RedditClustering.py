from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class PTT_RedditClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PTT_RedditClustering",
        description="Clustering of titles from 199 subreddits. Clustering of 25 sets, each with 10-50 classes, and each class with 100 - 1000 sentences.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "pt-mteb/translated_reddit-clustering",
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
        n_samples={"test": 420464},
        avg_character_length={"test": 64.7},
    )
