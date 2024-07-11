from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class PTT_SprintDuplicateQuestionsPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="PTT_SprintDuplicateQuestions",
        description="Duplicate questions from the Sprint community.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        dataset={
            "path": "pt-mteb/translated_sprintduplicatequestions-pairclassification",
            "revision": "main",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["validation", "test"],
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
        n_samples={"validation": 101000, "test": 101000},
        avg_character_length={"validation": 65.2, "test": 67.9},
    )
