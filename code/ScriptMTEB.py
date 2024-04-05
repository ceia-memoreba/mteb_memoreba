from sentence_transformers import SentenceTransformer
import logging

from mteb_local.evaluation.MTEB import MTEB

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEFAULT_PAD_TOKEN"] = "[PAD]"

logger = logging.getLogger(__name__)

# # Tasks

# ## Classification

# - Brazilian_court_decisionsClassification


class Brazilian_court_decisionsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "Brazilian_court_decisionsClassification",
            "hf_hub_name": "projetomemoreba/mteb_brazilian_court_decisions",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "ae5cdd58be9e246773486042e743abc6832906c8",
        }


# - HateBR_offensive_binary


class HateBR_offensive_binary_Classification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "HateBR_offensive_binary_Classification",
            "hf_hub_name": "projetomemoreba/mteb_HateBR_offensive_binary",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "1b392cfdfb8470a0b7067b2504a0f5ff79115f18",
        }


# - HateBR_offensive_levels


class HateBR_offensive_levels_Classification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "HateBR_offensive_levels_Classification",
            "hf_hub_name": "projetomemoreba/mteb_HateBR_offensive_levels",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "536bca26977850b9aa4d96099c5875d029c8c8fb",
        }


# - Portuguese_Hate_Speech_binary


class Portuguese_Hate_Speech_binary_Classification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "Portuguese_Hate_Speech_binary_Classification",
            "hf_hub_name": "projetomemoreba/mteb_Portuguese_Hate_Speech_binary",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "e4e57d33e334a8b73c4f7bc358f54b922a850b20",
        }


# - Told


class told_Classification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "told_Classification",
            "hf_hub_name": "projetomemoreba/mteb_told-br",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "b0129311d8a3e09ed458bcf325532a279b9cd8e7",
        }


# - legal_classification


class legal_Classification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "legal_Classification",
            "hf_hub_name": "projetomemoreba/mteb_memoreba_legal_classification",
            "description": (
                "A collection of Amazon reviews specifically designed to aid research in multilingual text"
                " classification."
            ),
            "reference": "https://arxiv.org/abs/2010.02573",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "accuracy",
            "revision": "82a34457c81aacb9dee488fcae1a2b5cdc9ab23d",
        }


# ## STS


class ASSIN2_STS(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "ASSIN2_STS",
            "hf_hub_name": "projetomemoreba/mteb_ASSIN2_STS",
            "description": "Semantic Textual Similarity Benchmark (STSbenchmark) dataset translated into German. "
            "Translations were originally done by T-Systems on site services GmbH.",
            "reference": "https://github.com/t-systems-on-site-services-gmbh/german-STSbenchmark",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "cosine_spearman",
            "min_score": 1,
            "max_score": 5,
            "revision": "3f42c21afdb9dbfa3f68ed21466ad92ed90a7b4b",
        }


# ## Retrieval


class faquad_Retrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "faquad_Retrieval",
            "hf_hub_name": "projetomemoreba/mteb_faquad",
            "description": "Test collection for passage retrieval from health-related Web resources in Spanish.",
            "reference": "https://mklab.iti.gr/results/spanish-passage-retrieval-dataset/",
            "type": "Retrieval",
            "category": "s2s",
            "eval_splits": ["validation", "test"],
            "eval_langs": ["pt"],
            "main_score": "ndcg_at_10",
            "revision": "279144718de5e6cb65d7d2b15968a393d33b41a8",
        }


# ## Reranking


class legal_reranking_content(AbsTaskReranking):
    @property
    def description(self):
        return {
            "name": "legal_reranking_content",

            "hf_hub_name": "projetomemoreba/mteb_memoreba_legal_reranking_content",
            "description": (
                "AskUbuntu Question Dataset - Questions from AskUbuntu with manual annotations marking pairs of"
                " questions as similar or non-similar"
            ),
            "reference": "https://github.com/taolei87/askubuntu",
            "type": "Reranking",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pt"],
            "main_score": "map",
            "revision": "11dd3cfadc389fad511a98a010a0f38815072a0e",
        }


# ## Clustering


class legal_clustering_s2s(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "legal_clustering_s2s",
            "hf_hub_name": "projetomemoreba/mteb_memoreba_legal_clustering_s2s",
            "description": (
                "Clustering of titles from arxiv. Clustering of 30 sets, either on the main or secondary category"
            ),
            "reference": "https://www.kaggle.com/Cornell-University/arxiv",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["pt"],
            "main_score": "v_measure",
            "revision": "9fd2db399e2a88887001cb88d5ac5a03cba36042",
        }

models = [
    "sentence-transformers/LaBSE",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/allenai-specter",
    "sentence-transformers/average_word_embeddings_glove.6B.300d",
    "sentence-transformers/average_word_embeddings_komninos",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "sentence-transformers/gtr-t5-base",
    "sentence-transformers/gtr-t5-large",
    "sentence-transformers/gtr-t5-xl",
    "sentence-transformers/gtr-t5-xxl",
    "sentence-transformers/msmarco-bert-co-condensor",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/sentence-t5-base",
    "sentence-transformers/sentence-t5-large",
    "sentence-transformers/sentence-t5-xl",
    "sentence-transformers/sentence-t5-xxl",
    "intfloat/e5-base",
    "intfloat/e5-base-v2",
    "intfloat/e5-large",
    "intfloat/e5-large-v2",
    "intfloat/e5-mistral-7b-instruct",
    "intfloat/e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
    "intfloat/multilingual-e5-small"
]

models_multilingual = [
 'sentence-transformers/LaBSE',
 'sentence-transformers/allenai-specter',
 'sentence-transformers/average_word_embeddings_glove.6B.300d',
 'sentence-transformers/average_word_embeddings_komninos',
 'sentence-transformers/distiluse-base-multilingual-cased-v2',
 'sentence-transformers/msmarco-bert-co-condensor',
 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
 'intfloat/multilingual-e5-base',
 'intfloat/multilingual-e5-large',
 'intfloat/multilingual-e5-large-instruct',
 'intfloat/multilingual-e5-small'
]

models_multilingual_v2 = [
   'FacebookAI/xlm-roberta-base',
   'google-bert/bert-base-uncased',
   'Jechto/e5-dansk-test-0.1',
   'mukaj/fin-mpnet-base',
   'nthakur/contriever-base-msmarco',
   'princeton-nlp/unsup-simcse-bert-base-uncased',
   'princeton-nlp/sup-simcse-bert-base-uncased',
   'shibing624/text2vec-base-multilingual',
   'silk-road/luotuo-bert-medium',
   'thtang/ALL_862873'
]

models_multilingual_v3 = [
    'Salesforce/SFR-Embedding-Mistral',
    'GritLM/GritLM-7B',
    'intfloat/e5-mistral-7b-instruct',
    'GritLM/GritLM-8x7B',
    'jspringer/echo-mistral-7b-instruct-lasttoken',
    'mixedbread-ai/mxbai-embed-large-v1',
    'mixedbread-ai/mxbai-embed-2d-large-v1',
    'WhereIsAI/UAE-Large-V1',
    'Cohere/Cohere-embed-english-v3.0',
    'intfloat/multilingual-e5-large-instruct',
    'avsolatorio/GIST-large-Embedding-v0',
    'BAAI/bge-large-en-v1.5',
    'llmrails/ember-v1',
    'jamesgpt1/sf_model_e5',
    'thenlper/gte-large',
    'avsolatorio/GIST-small-Embedding-v0',
    'infgrad/stella-base-en-v2',
    'thenlper/gte-base',
    'nomic-ai/nomic-embed-text-v1',
    'nomic-ai/nomic-embed-text-v1.5',
    'intfloat/e5-large-v2',
    'BAAI/bge-small-en-v1.5',
    'Cohere/Cohere-embed-english-light-v3.0',
    'nomic-ai/nomic-embed-text-v1.5',
    'hkunlp/instructor-xl',
    'hkunlp/instructor-large',
    'intfloat/multilingual-e5-large',
    'izhx/udever-bloom-3b'
    'nomic-ai/nomic-embed-text-v1-ablated',
    'thenlper/gte-small',
    'nomic-ai/nomic-embed-text-v1.5',
    'avsolatorio/GIST-all-MiniLM-L6-v2',
    'hkunlp/instructor-large',
    'intfloat/e5-base-v2'
]

models_multilingual_v4 = [
    'GritLM/GritLM-7B',
    'GritLM/GritLM-8x7B'
]

models_multilingual_v5 = [
    'intfloat/e5-large',
    'nomic-ai/nomic-embed-text-v1-ablated',
    'thenlper/gte-small',
    'nomic-ai/nomic-embed-text-v1.5',
    'izhx/udever-bloom-7b1',
    'intfloat/e5-base',
    'jinaai/jina-embeddings-v2-base-en',
    'Cohere/Cohere-embed-multilingual-light-v3.0',
    'intfloat/e5-small-v2',
    'izhx/udever-bloom-3b',
    'nomic-ai/nomic-embed-text-v1-unsupervised',
    'intfloat/multilingual-e5-large'
]

models_multilingual_v6 = [
    'zhou-xl/bi-cse',
    'zeroshot/gte-small-quant',
    'zeroshot/gte-large-sparse',
    'neuralmagic/bge-base-en-v1.5-quant',
    'mgoin/all-MiniLM-L6-v2-ds',
    'neuralmagic/bge-large-en-v1.5-quant',
    'neuralmagic/bge-base-en-v1.5-sparse',
    'neuralmagic/bge-small-en-v1.5-sparse',
    'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'jinaai/jina-embeddings-v2-base-de',
    'jinaai/jina-embeddings-v2-base-es',
    'consciousAI/cai-lunaris-text-embeddings',
    'consciousAI/cai-stellaris-text-embeddings',
    'aws-neuron/bge-base-en-v1-5-seqlen-384-bs-1',
    'Shimin/yiyouliao',
    'Shimin/LLaMA-embeeding',
    'ManiShankar-AlpesAi/paraphrase-multilingual-mpnet-base-v2-KE_Sieve'
]

TASK_LIST_CLASSIFICATION = [
    "Brazilian_court_decisionsClassification",
    "HateBR_offensive_binary_Classification",
    "HateBR_offensive_levels_Classification",
    "Portuguese_Hate_Speech_binary_Classification",
    "told_Classification",
    "legal_Classification",
]

TASK_LIST_CLUSTERING = ["legal_clustering_s2s"]

TASK_LIST_RERANKING = ["legal_reranking_content"]

TASK_LIST_RETRIEVAL = ["faquad_Retrieval"]

TASK_LIST_STS = ["ASSIN2_STS"]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_name in models_multilingual_v6:
    try:
        model = SentenceTransformer(model_name)
        model.to(device)
    except Exception as e:
        logger.error(f"Error while evaluating model {model_name}: {e}")
    else:
        evaluation = MTEB(tasks=TASK_LIST, task_langs=["pt"])
        evaluation.run(model, model_name, overwrite_results=True, output_folder=f"results/{model_name}")