from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class MilkQAReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="MilkQA-Reranking",
        description="MilkQA is a dataset of dense questions for the task of answer selection. It contains questions and answers of the dairy farming domain that were collected by the customer service of Embrapa Dairy Cattle between the years of 2003 and 2012.",
        reference="https://arxiv.org/abs/1801.03460",
        dataset={
            "path": "mteb-pt/milkqa-reranking",
            "revision": "ea111dd108c123e055854b286e97177f04507818",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["dev", "test"],
        eval_langs=["por-Latn"],
        main_score="map",
        date=("2003-01-01", "2012-12-31"),
        form=["written"],
        domains=[],
        task_subtypes=[],
        license="cc-by-nc-nd-4.0",
        socioeconomic_status=None,
        annotations_creators="human-annotated",
        dialect=["pt-BR"],
        text_creation="found",
        bibtex_citation="""
@inproceedings{criscuolo2017milkqa,
    author = {Marcelo Criscuolo and Erick Rocha Fonseca and Sandra Maria Aluísio and Ana Carolina Sperança-Criscuolo},
    title = {{MilkQA}: a Dataset of Consumer Questions for the Task of Answer Selection},
    booktitle = {Proceedings of the 6th Brazilian Conference on Intelligent Systems (BRACIS)},
    year = {2017},
    month = {October},
    date = {2-5},
    address = {Uberlândia, Brazil},
    publisher = {IEEE},
    isbn = {978-1-5386-2407-4},
    pages = {354--359},
    volume = {1},
    doi = {10.1109/BRACIS.2017.12},
}
""",
    n_samples={'dev': 50, 'test': 300},
    avg_character_length={'dev': 2553.2, 'test': 2586.9}
    )
