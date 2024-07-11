from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class FactNewsFactualityClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="FactNewsFactuality",
        description="FactNews is large sentence-level dataset composed of 6,191 sentences expertly annotated according to factuality and media bias definitions proposed by AllSides.",
        reference="https://aclanthology.org/2023.ranlp-1.127",
        dataset={
            "path": "eduagarcia/FactNews",
            "name": "factuality_prediction",
            "revision": "683cbf3c4179cedde3a5f6df5f189ac7a41591ce",
        },
        type="Classification",
        category="s2s",
        date=("2006-01-01", "2023-01-01"),
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="f1",
        form=["written"],
        domains=["News"],
        task_subtypes=["Political classification"],
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{vargas-etal-2023-predicting,
    title = "Predicting Sentence-Level Factuality of News and Bias of Media Outlets",
    author = "Vargas, Francielle  and
      Jaidka, Kokil  and
      Pardo, Thiago  and
      Benevenuto, Fabr{\'\i}cio",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ranlp-1.127",
    pages = "1197--1206",
}
""",
        n_samples={'train': 2826, 'test': 1788},
        avg_character_length={'train': 119.9, 'test': 119.9}
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentences", "text")
