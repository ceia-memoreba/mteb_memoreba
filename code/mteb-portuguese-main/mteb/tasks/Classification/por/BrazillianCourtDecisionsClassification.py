from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class BrazillianCourtDecisionsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="brazillian_court_decisions_judgment",
        description="A collection of Ementas (summary) court decisions and their metadata from the Tribunal de Justiça de Alagoas (TJAL, the State Supreme Court of Alagoas (Brazil). The court decisions are labeled according to 3 categories.",
        reference="https://arxiv.org/abs/1905.10348",
        dataset={
            "path": "joelniklaus/brazilian_court_decisions",
            "revision": "e937c2db8eab109cafc4f5279a396957d38251c5",
        },
        type="Classification",
        category="s2s",
        date=("2019-01-01", "2020-01-01"),
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="f1",
        form=["written"],
        domains=["Legal"],
        task_subtypes=[],
        license=None,
        socioeconomic_status=None,
        annotations_creators="derived",
        dialect=["pt-BR"],
        text_creation="found",
        bibtex_citation="""
@article{Lage-Freitas2022,
  author = {Lage-Freitas, Andr{\'{e}} and Allende-Cid, H{\'{e}}ctor and Santana, Orivaldo and Oliveira-Lage, L{\'{i}}via},
  doi = {10.7717/peerj-cs.904},
  issn = {2376-5992},
  journal = {PeerJ. Computer science},
  keywords = {Artificial intelligence,Jurimetrics,Law,Legal,Legal NLP,Legal informatics,Legal outcome forecast,Litigation prediction,Machine learning,NLP,Portuguese,Predictive algorithms,judgement prediction},
  language = {eng},
  month = {mar},
  pages = {e904--e904},
  publisher = {PeerJ Inc.},
  title = {{Predicting Brazilian Court Decisions}},
  url = {https://pubmed.ncbi.nlm.nih.gov/35494851 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044329/},
  volume = {8},
  year = {2022}
}
""",
        n_samples={"train": 3234, "validation": 404, "test": 405},
        avg_character_length=None,
    )

    def dataset_transform(self):
        map_labels = {"no": 0, "partial": 1, "yes": 2}
        self.dataset = self.dataset.map(
            lambda example: {
                "text": example["decision_description"],
                "label": map_labels[example["judgment_label"]],
            }
        )
        self.dataset = self.dataset.remove_columns(
            [col for col in self.dataset[self.metadata.eval_splits[0]].column_names if col not in ["text", "label"]]
        )
