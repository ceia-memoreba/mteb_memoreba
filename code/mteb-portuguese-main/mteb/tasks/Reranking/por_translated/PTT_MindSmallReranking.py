from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class PTT_MindSmallReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="PTT_MindSmallReranking",
        description="Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research",
        reference="https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
        hf_hub_name="mteb/mind_small",
        dataset={
            "path": "pt-mteb/translated_mind_small",
            "revision": "main",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="map",
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
        n_samples={"test": 107968},
        avg_character_length={"test": 70.9},
    )
