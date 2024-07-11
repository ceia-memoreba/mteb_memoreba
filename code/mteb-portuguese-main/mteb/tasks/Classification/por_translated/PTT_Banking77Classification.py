from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class PTT_Banking77Classification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PTT_Banking77Classification",
        description="Dataset composed of online banking queries annotated with their corresponding intents.",
        reference="https://arxiv.org/abs/2003.04807",
        dataset={
            "path": "pt-mteb/translated_banking77",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
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
        n_samples={"test": 3080},
        avg_character_length={"test": 54.2},
    )
