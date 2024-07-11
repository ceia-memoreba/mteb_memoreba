from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

class AssinSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="ASSIN-STS",
        dataset={
            "path": "assin",
            "name": "full",
            "revision": "6535e48351178e07ade013b05b69f0e35cb28bbb",
        },
        description="The ASSIN (Avaliação de Similaridade Semântica e Inferência textual) corpus is a corpus annotated with pairs of sentences written in Portuguese that is suitable for the exploration of textual entailment and paraphrasing classifiers. ",
        reference="https://ceur-ws.org/Vol-2583/1_ASSIN-2.pdf", #http://nilc.icmc.usp.br/assin/
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
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
        n_samples={"test": 4000},
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