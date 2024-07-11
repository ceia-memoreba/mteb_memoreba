from __future__ import annotations

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class PTT_BigPatentClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PTT_BigPatentClustering",
        description="Clustering of documents from the Big Patent dataset. Test set only includes documents"
        "belonging to a single category, with a total of 9 categories.",
        reference="https://www.kaggle.com/datasets/big_patent",
        dataset={
            "path": "pt-mteb/translated_big-patent-clustering",
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
        n_samples=None,
        avg_character_length=None,
    )
