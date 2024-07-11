from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

class Assin2STS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="ASSIN2-STS",
        dataset={
            "path": "assin2",
            "revision": "0ff9c86779e06855536d8775ce5550550e1e5a2d",
        },
        description="The ASSIN 2 corpus is composed of rather simple sentences. Following the procedures of SemEval 2014 Task 1. The training and validation data are composed, respectively, of 6,500 and 500 sentence pairs in Brazilian Portuguese, annotated for entailment and semantic similarity. ",
        reference="https://ceur-ws.org/Vol-2583/1_ASSIN-2.pdf", #https://sites.google.com/view/assin2/
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=("2020-01-01", "2021-01-01"),  # rough approximates
        form=["written"],
        domains=["Non-fiction"],
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        text_creation="machine-translated and localized",
        bibtex_citation="""
@inproceedings{real2020assin,
  title={The assin 2 shared task: a quick overview},
  author={Real, Livy and Fonseca, Erick and Oliveira, Hugo Goncalo},
  booktitle={International Conference on Computational Processing of the Portuguese Language},
  pages={406--412},
  year={2020},
  organization={Springer}
}
""",
        n_samples={"test": 2448},
        avg_character_length=None,
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 1
        metadata_dict["max_score"] = 5
        return metadata_dict
    
    def dataset_transform(self):
        map_names = {
            "premise": "sentence1",
            "hypothesis": "sentence2",
            "relatedness_score": "score",
            "sentence_pair_id": "id",
        }
        self.dataset = self.dataset.remove_columns(
            [col for col in self.dataset[self.metadata.eval_splits[0]].column_names if col not in map_names]
        )
        self.dataset = self.dataset.rename_columns(map_names)
    

