from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class PTT_STSBenchmarkSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="PTT_STSBenchmark",
        dataset={
            "path": "pt-mteb/translated_stsbenchmark-sts",
            "revision": "main",
        },
        description="Semantic Textual Similarity Benchmark (STSbenchmark) dataset.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
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

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
