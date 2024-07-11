from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification

class Assin2RtePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ASSIN2-RTE",
        dataset={
            "path": "assin2",
            "revision": "0ff9c86779e06855536d8775ce5550550e1e5a2d",
        },
        description="The ASSIN 2 corpus is composed of rather simple sentences. Following the procedures of SemEval 2014 Task 1. The training and validation data are composed, respectively, of 6,500 and 500 sentence pairs in Brazilian Portuguese, annotated for entailment and semantic similarity. ",
        reference="https://ceur-ws.org/Vol-2583/1_ASSIN-2.pdf", #https://sites.google.com/view/assin2/
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="f1",
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
    
    def dataset_transform(self):
        _dataset = {}
        for split in self.metadata.eval_splits:
            hf_dataset = self.dataset[split]

            _dataset[split] = [
                {
                    "sent1": hf_dataset["premise"],
                    "sent2": hf_dataset["hypothesis"],
                    "labels": hf_dataset["entailment_judgment"],
                }
            ]
        self.dataset = _dataset
    

