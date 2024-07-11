from __future__ import annotations

from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class ToLDBrClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="ToLD-Br",
        description="ToLD-Br is the biggest dataset for toxic tweets in Brazilian Portuguese, crowdsourced by 42 annotators selected from a pool of 129 volunteers.",
        reference="https://arxiv.org/abs/2010.04543",
        dataset={
            "path": "told-br",
            "revision": "fb4f11a5bc68b99891852d20f1ec074be6289768",
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
@article{DBLP:journals/corr/abs-2010-04543,
  author    = {Joao Augusto Leite and
               Diego F. Silva and
               Kalina Bontcheva and
               Carolina Scarton},
  title     = {Toxic Language Detection in Social Media for Brazilian Portuguese:
               New Dataset and Multilingual Analysis},
  journal   = {CoRR},
  volume    = {abs/2010.04543},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.04543},
  eprinttype = {arXiv},
  eprint    = {2010.04543},
  timestamp = {Tue, 15 Dec 2020 16:10:16 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-04543.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
""",
        n_samples={"train": 3969, "validation": 850, "test": 851},
        avg_character_length=None,
    )
