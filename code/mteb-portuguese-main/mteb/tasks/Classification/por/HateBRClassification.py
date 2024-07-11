from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HateBRClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HateBR",
        description="HateBR is the first large-scale expert annotated dataset of Brazilian Instagram comments for abusive language detection on the web and social media.",
        reference="https://aclanthology.org/2022.lrec-1.777/",
        dataset={
            "path": "eduagarcia/portuguese_benchmark",
            "name": "HateBR_offensive_level",
            "revision": "5b9a2bcca49a130aa08fc7c788c4c74441d4aae3",
        },
        type="Classification",
        category="s2s",
        date=("2022-01-01", "2023-01-01"),
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="f1",
        form=["written"],
        domains=["Social", "Web"],
        task_subtypes=["Sentiment/Hate speech"],
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{vargas2022hatebr,
  title={HateBR: A Large Expert Annotated Corpus of Brazilian Instagram Comments for Offensive Language and Hate Speech Detection},
  author={Vargas, Francielle and Carvalho, Isabelle and de G{\'o}es, Fabiana Rodrigues and Pardo, Thiago and Benevenuto, Fabr{\'\i}cio},
  booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  pages={7174--7183},
  year={2022}
}
""",
        n_samples={"train": 4480, "validation": 1120, "test": 1400},
        avg_character_length=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")
