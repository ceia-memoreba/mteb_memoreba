from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class PTT_LegalBenchConsumerContractsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="PTT_LegalBenchConsumerContractsQA",
        description="The dataset includes questions and answers related to contracts.",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa",
        dataset={
            "path": "pt-mteb/translated_legalbench_consumer_contracts_qa",
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
        task_subtypes=["Question answering"],
        license="CC BY-NC 4.0",
        socioeconomic_status="high",
        annotations_creators="derived",
        dialect=None,
        text_creation="found",
        bibtex_citation=None,
        n_samples=None,
        avg_character_length=None,
    )
