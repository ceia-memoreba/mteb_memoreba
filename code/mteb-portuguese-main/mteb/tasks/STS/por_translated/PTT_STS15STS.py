from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class PTT_STS15STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="PTT_STS15",
        dataset={
            "path": "pt-mteb/translated_sts15-sts",
            "revision": "main",
        },
        description="SemEval STS 2015 dataset",
        reference="https://www.aclweb.org/anthology/S15-2010",
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
