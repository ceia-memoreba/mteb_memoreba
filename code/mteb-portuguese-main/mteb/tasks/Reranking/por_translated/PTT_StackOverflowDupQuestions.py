from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class PTT_StackOverflowDupQuestions(AbsTaskReranking):
    metadata = TaskMetadata(
        name="PTT_StackOverflowDupQuestions",
        description="Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python",
        reference="https://www.microsoft.com/en-us/research/uploads/prod/2019/03/nl4se18LinkSO.pdf",
        dataset={
            "path": "pt-mteb/translated_stackoverflowdupquestions-reranking",
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
        n_samples={"test": 3467},
        avg_character_length={"test": 49.8},
    )
