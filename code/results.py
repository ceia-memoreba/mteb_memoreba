"""MTEB Results"""

import json

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """@article{muennighoff2022mteb,
  doi = {10.48550/ARXIV.2210.07316},
  url = {https://arxiv.org/abs/2210.07316},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},  
  year = {2022}
}
"""

_DESCRIPTION = """Results on MTEB Portuguese"""

URL = "https://huggingface.co/datasets/projetomemoreba/results/resolve/main/paths.json"
VERSION = datasets.Version("1.0.1")
EVAL_LANGS = ['pt']

SKIP_KEYS = ["std", "evaluation_time", "main_score", "threshold"]

# Use "train" split instead
TRAIN_SPLIT = ["DanishPoliticalCommentsClassification"]
# Use "validation" split instead
VALIDATION_SPLIT = ["AFQMC", "Cmnli", "IFlyTek", "TNews", "MSMARCO", "MSMARCO-PL", "MultilingualSentiment", "Ocnli"]
# Use "dev" split instead
DEV_SPLIT = ["CmedqaRetrieval", "CovidRetrieval", "DuRetrieval", "EcomRetrieval", "MedicalRetrieval", "MMarcoReranking", "MMarcoRetrieval", "MSMARCO", "MSMARCO-PL", "T2Reranking", "T2Retrieval", "VideoRetrieval"]



MODELS = [
    "ALL_862873",
    "allenai-specter",
    "average_word_embeddings_glove.6B.300d",
    "average_word_embeddings_komninos",
    "bert-base-15lang-cased",
    "bert-base-10lang-cased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-portuguese-cased",
    "bert-base-uncased",
    "contriever-base-msmarco",
    "dfm-sentence-encoder-large-v1",
    "distilbert-base-25lang-cased",
    "distilbert-base-en-fr-cased",
    "distilbert-base-en-fr-es-pt-it-cased",
    "distilbert-base-fr-cased",
    "distilbert-base-uncased",
    "distiluse-base-multilingual-cased-v2",
    "e5-dansk-test-0.1",
    "fin-mpnet-base",
    "LaBSE",
    "luotuo-bert-medium",
    "msmarco-bert-co-condensor",
    "multilingual-e5-base",
    "multilingual-e5-large",
    "multilingual-e5-large-instruct",
    "multilingual-e5-small",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "sup-simcse-bert-base-uncased",
    "text2vec-base-multilingual",
    "unsup-simcse-bert-base-uncased",
    "xlm-roberta-base",
    "xlm-roberta-large"
]

from pathlib import Path

# Needs to be run whenever new files are added
def get_paths():
    import collections, json, os
    files = collections.defaultdict(list)
    for model_dir in os.listdir("results"):
        results_model_dir = os.path.join("results", model_dir)
        if not os.path.isdir(results_model_dir):
            print(f"Skipping {results_model_dir}")
            continue
        for res_file in os.listdir(results_model_dir):
            if res_file.endswith(".json"):
                results_model_file = os.path.join(results_model_dir, res_file)
                files[model_dir].append(results_model_file)
            
    with open("paths.json", "w") as f:
        json.dump(files, f)
    return files

class MTEBResults(datasets.GeneratorBasedBuilder):
    """MTEBResults"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=model,
            description=f"{model} MTEB results",
            version=VERSION,
        )
        for model in MODELS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "mteb_dataset_name": datasets.Value("string"),
                    "eval_language": datasets.Value("string"),
                    "metric": datasets.Value("string"),
                    "score": datasets.Value("float"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        path_file = dl_manager.download_and_extract(URL)
        with open(path_file) as f:
            files = json.load(f)

        downloaded_files = dl_manager.download_and_extract(files[self.config.name])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={'filepath': downloaded_files}
            )
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info(f"Generating examples from {filepath}")

        out = []

        for path in filepath:
            with open(path, encoding="utf-8") as f:
                res_dict = json.load(f)
                ds_name = res_dict["mteb_dataset_name"]
                split = "test"
                if (ds_name in TRAIN_SPLIT) and ("train" in res_dict):
                    split = "train"
                elif (ds_name in VALIDATION_SPLIT) and ("validation" in res_dict):
                    split = "validation"
                elif (ds_name in DEV_SPLIT) and ("dev" in res_dict):
                    split = "dev"
                elif "test" not in res_dict:
                    print(f"Skipping {ds_name} as split {split} not present.")
                    continue
                res_dict = res_dict.get(split)
                is_multilingual = any(x in res_dict for x in EVAL_LANGS)
                langs = res_dict.keys() if is_multilingual else ["en"]
                for lang in langs:
                    if lang in SKIP_KEYS: continue
                    test_result_lang = res_dict.get(lang) if is_multilingual else res_dict
                    for metric, score in test_result_lang.items():
                        if not isinstance(score, dict):
                            score = {metric: score}
                        for sub_metric, sub_score in score.items():
                            if any(x in sub_metric for x in SKIP_KEYS): continue
                            out.append({
                                "mteb_dataset_name": ds_name,
                                "eval_language": lang if is_multilingual else "",
                                "metric": f"{metric}_{sub_metric}" if metric != sub_metric else metric,
                                "score": sub_score * 100,
                            })
        for idx, row in enumerate(sorted(out, key=lambda x: x["mteb_dataset_name"])):
            yield idx, row
