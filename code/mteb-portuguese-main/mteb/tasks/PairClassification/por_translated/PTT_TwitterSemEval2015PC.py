from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PTT_TwitterSemEval2015PC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PTT_TwitterSemEval2015",
        dataset={
            "path": "pt-mteb/translated_twittersemeval2015-pairclassification",
            "revision": "main",
        },
        description="Paraphrase-Pairs of Tweets from the SemEval 2015 workshop.",
        reference="https://alt.qcri.org/semeval2015/task1/",
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
        n_samples={"test": 16777},
        avg_character_length={"test": 38.3},
    )
