from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class PTT_AskUbuntuDupQuestions(AbsTaskReranking):
    metadata = TaskMetadata(
        name="PTT_AskUbuntuDupQuestions",
        description="AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar",
        reference="https://github.com/taolei87/askubuntu",
        dataset={
            "path": "pt-mteb/translated_askubuntudupquestions-reranking",
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
        n_samples={"test": 2255},
        avg_character_length={"test": 52.5},
    )
