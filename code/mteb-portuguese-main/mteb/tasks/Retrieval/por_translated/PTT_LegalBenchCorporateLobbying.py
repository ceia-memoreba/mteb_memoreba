from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_LegalBenchCorporateLobbying(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_LegalBenchCorporateLobbying",
        description="The dataset includes bill titles and bill summaries related to corporate lobbying.",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying",
        dataset={
            "path": "pt-mteb/translated_legalbench_corporate_lobbying",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="ndcg_at_10",
        date=None,
        form=["written"],
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="CC BY 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
