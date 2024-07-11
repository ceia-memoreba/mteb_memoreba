from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class PortugueseHateSpeechClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PortugueseHateSpeech",
        description="The dataset is composed of 5,668 tweets. For its annotation, we defined two different schemes used by annotators with different levels of expertise. Firstly, non-experts annotated the tweets with binary labels (‘hate’ vs. ‘no-hate’).",
        reference="https://aclanthology.org/W19-3510/",
        dataset={
            "path": "eduagarcia/portuguese_benchmark",
            "name": "Portuguese_Hate_Speech_binary",
            "revision": "5b9a2bcca49a130aa08fc7c788c4c74441d4aae3",
        },
        type="Classification",
        category="s2s",
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="f1",
        form=["written"],
        domains=["Social"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status=None,
        annotations_creators="human-annotated",
        dialect=["pt-BR"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{fortuna-etal-2019-hierarchically,
    title = "A Hierarchically-Labeled {P}ortuguese Hate Speech Dataset",
    author = "Fortuna, Paula  and
      Rocha da Silva, Jo{\~a}o  and
      Soler-Company, Juan  and
      Wanner, Leo  and
      Nunes, S{\'e}rgio",
    editor = "Roberts, Sarah T.  and
      Tetreault, Joel  and
      Prabhakaran, Vinodkumar  and
      Waseem, Zeerak",
    booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-3510",
    doi = "10.18653/v1/W19-3510",
    pages = "94--104",
}
""",
        n_samples={"train": 3969, "validation": 850, "test": 851},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
