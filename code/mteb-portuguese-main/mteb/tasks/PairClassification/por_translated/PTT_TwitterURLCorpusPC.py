from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PTT_TwitterURLCorpusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PTT_TwitterURLCorpus",
        dataset={
            "path": "pt-mteb/translated_twitterurlcorpus-pairclassification",
            "revision": "main",
        },
        description="Paraphrase-Pairs of Tweets.",
        reference="https://languagenet.github.io/",
        category="s2s",
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ap",
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
        n_samples={"test": 51534},
        avg_character_length={"test": 79.5},
    )
