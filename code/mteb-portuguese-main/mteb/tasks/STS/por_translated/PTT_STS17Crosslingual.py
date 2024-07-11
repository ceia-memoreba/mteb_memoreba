from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskSTS, CrosslingualTask

_LANGUAGES = {
    "ko-ko": ["kor-Hang"],
    "ar-ar": ["ara-Arab"],
    "pt-ar": ["por-Latn", "ara-Arab"],
    "pt-de": ["por-Latn", "deu-Latn"],
    "pt-pt": ["por-Latn"],
    "pt-tr": ["por-Latn", "tur-Latn"],
    "es-pt": ["spa-Latn", "por-Latn"],
    "es-es": ["spa-Latn"],
    "fr-pt": ["fra-Latn", "por-Latn"],
    "it-pt": ["ita-Latn", "por-Latn"],
    "nl-pt": ["nld-Latn", "por-Latn"],
}


class PTT_STS17Crosslingual(AbsTaskSTS, CrosslingualTask):
    metadata = TaskMetadata(
        name="PTT_STS17",
        dataset={
            "path": "pt-mteb/translated_sts17-crosslingual-sts",
            "revision": "main",
        },
        description="STS 2017 dataset",
        reference="http://alt.qcri.org/semeval2016/task1/",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
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
        n_samples={"test": 500},
        avg_character_length={"test": 43.3},
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
