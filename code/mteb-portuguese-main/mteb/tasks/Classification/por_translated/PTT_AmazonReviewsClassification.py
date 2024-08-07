from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification, MultilingualTask


class PTT_AmazonReviewsClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="PTT_AmazonReviewsClassification",
        dataset={
            "path": "pt-mteb/translated_amazon_reviews_multi",
            "revision": "main",
        },
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://arxiv.org/abs/2010.02573",
        category="s2s",
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={
            "pt": ["por-Latn"],
            "de": ["deu-Latn"],
            "es": ["spa-Latn"],
            "fr": ["fra-Latn"],
            "ja": ["jpn-Jpan"],
            "zh": ["cmn-Hans"],
        },
        main_score="accuracy",
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
        n_samples={"validation": 30000, "test": 30000},
        avg_character_length={"validation": 159.2, "test": 160.4},
    )
