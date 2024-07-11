from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_HotpotQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_HotpotQA",
        dataset={
            "path": "pt-mteb/translated_hotpotqa",
            "revision": "main",
        },
        description=(
            "PTT_HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong"
            " supervision for supporting facts to enable more explainable question answering systems."
        ),
        reference="https://hotpotqa.github.io/",
        type="Retrieval",
        category="s2p",
        eval_splits=["train", "dev", "test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
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
