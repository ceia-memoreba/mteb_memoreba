from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

class SickBrSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="SICK-BR-STS",
        dataset={
            "path": "eduagarcia/sick-br",
            "revision": "0cdfb1d51ef339011c067688a3b75b82f927c097",
        },
        description="A Portuguese inference corpus, aligned to and translated from SICK. Created by machine translation and human post editing of the SICK dataset.",
        reference="https://linux.ime.usp.br/~thalen/SICK_PT.pdf9",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["por-Latn"],
        main_score="cosine_spearman",
        date=("2018-01-01", "2019-01-01"),  # rough approximates
        form=["written"],
        domains=["Non-fiction"],
        task_subtypes=None,
        license=None,
        socioeconomic_status=None,
        annotations_creators="expert-annotated",
        dialect=["pt-BR"],
        text_creation="machine-translated and localized",
        bibtex_citation="""
@inproceedings{real18,
  author="Real, Livy
    and Rodrigues, Ana
    and Vieira e Silva, Andressa
    and Albiero, Beatriz
    and Thalenberg, Bruna
    and Guide, Bruno
    and Silva, Cindy
    and de Oliveira Lima, Guilherme
    and C{\^a}mara, Igor C. S.
    and Stanojevi{\'{c}}, Milo{\v{s}}
    and Souza, Rodrigo
    and de Paiva, Valeria"
  year ="2018",
  title="SICK-BR: A Portuguese Corpus for Inference",
  booktitle="Computational Processing of the Portuguese Language. PROPOR 2018.",
  doi ="10.1007/978-3-319-99722-3_31",
  isbn="978-3-319-99722-3"
}
""",
        n_samples={"test": 4906},
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
            "sentence_A": "sentence1",
            "sentence_B": "sentence2",
            "relatedness_score": "score",
            "pair_ID": "id",
        }
        self.dataset = self.dataset.remove_columns(
            [col for col in self.dataset[self.metadata.eval_splits[0]].column_names if col not in map_names]
        )
        self.dataset = self.dataset.rename_columns(map_names)
        

