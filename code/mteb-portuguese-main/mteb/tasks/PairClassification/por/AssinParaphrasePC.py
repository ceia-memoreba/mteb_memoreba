from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification

class AssinParaphrasePC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="ASSIN-paraphrase",
        dataset={
            "path": "assin",
            "name": "full",
            "revision": "6535e48351178e07ade013b05b69f0e35cb28bbb",
        },
        description="The ASSIN (Avaliação de Similaridade Semântica e Inferência textual) corpus is a corpus annotated with pairs of sentences written in Portuguese that is suitable for the exploration of textual entailment and paraphrasing classifiers. ",
        reference="https://ceur-ws.org/Vol-2583/1_ASSIN-2.pdf", #http://nilc.icmc.usp.br/assin/
        type="PairClassification",
        category="s2s",
        eval_splits=["train", "validation", "test"],
        eval_langs=["por-Latn"],
        main_score="ap",
        date=("2016-01-01", "2017-01-01"),  # rough approximates
        form=["written"],
        domains=["Non-fiction", "News"],
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR", "pt-PT"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{fonseca2016assin,
  title={ASSIN: Avaliacao de similaridade semantica e inferencia textual},
  author={Fonseca, E and Santos, L and Criscuolo, Marcelo and Aluisio, S},
  booktitle={Computational Processing of the Portuguese Language-12th International Conference, Tomar, Portugal},
  pages={13--15},
  year={2016}
}
""",
        n_samples={"test": 4000, "validation": 1000, "train": 5000},
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
                    "labels": [int(l==2) for l in hf_dataset["entailment_judgment"]],
                }
            ]
        self.dataset = _dataset