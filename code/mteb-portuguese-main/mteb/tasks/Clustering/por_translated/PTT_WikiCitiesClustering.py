from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskClustering import AbsTaskClustering


class PTT_WikiCitiesClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="PTT_WikiCitiesClustering",
        description="Clustering of Wikipedia articles of cities by country from https://huggingface.co/datasets/wikipedia. Test set includes 126 countries, and a total of 3531 cities.",
        reference="https://huggingface.co/datasets/wikipedia",
        dataset={
            "path": "pt-mteb/translated_cities_wiki_clustering",
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
        n_samples=None,
        avg_character_length=None,
    )
