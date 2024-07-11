"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging
import gc
import torch

from sentence_transformers import SentenceTransformer

from mteb import MTEB
import traceback

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("output.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")


######################################

TASK_LIST_TRANSLATED_CLASSIFICATION = [
    "PTT_AmazonCounterfactualClassification",
    "PTT_AmazonPolarityClassification",
    "PTT_AmazonReviewsClassification",
    "PTT_Banking77Classification",
    "PTT_EmotionClassification",
    "PTT_ImdbClassification",
    #"PTT_MassiveIntentClassification",
    #"PTT_MassiveScenarioClassification",
    "PTT_MTOPDomainClassification",
    "PTT_MTOPIntentClassification",
    "PTT_ToxicConversationsClassification",
    "PTT_TweetSentimentExtractionClassification",
]

TASK_LIST_TRANSLATED_CLUSTERING = [
    "PTT_ArxivClusteringP2P",
    "PTT_ArxivClusteringS2S",
    "PTT_BiorxivClusteringP2P",
    "PTT_BiorxivClusteringS2S",
    "PTT_MedrxivClusteringP2P",
    "PTT_MedrxivClusteringS2S",
    "PTT_RedditClustering",
    "PTT_RedditClusteringP2P",
    "PTT_StackExchangeClustering",
    "PTT_StackExchangeClusteringP2P",
    "PTT_TwentyNewsgroupsClustering",
]

TASK_LIST_TRANSLATED_PAIR_CLASSIFICATION = [
    "PTT_SprintDuplicateQuestions",
    "PTT_TwitterSemEval2015",
    "PTT_TwitterURLCorpus",
]

TASK_LIST_TRANSLATED_RERANKING = [
    "PTT_AskUbuntuDupQuestions",
    "PTT_MindSmallReranking",
    "PTT_SciDocsRR",
    "PTT_StackOverflowDupQuestions",
]

TASK_LIST_TRANSLATED_RETRIEVAL = [
    "PTT_ArguAna",
    "PTT_ClimateFEVER",
    #"PTT_CQADupstackAndroidRetrieval",
    #"PTT_CQADupstackEnglishRetrieval",
    #"PTT_CQADupstackGamingRetrieval",
    #"PTT_CQADupstackGisRetrieval",
    #"PTT_CQADupstackMathematicaRetrieval",
    #"PTT_CQADupstackPhysicsRetrieval",
    #"PTT_CQADupstackProgrammersRetrieval",
    "PTT_CQADupstackStatsRetrieval",
    #"PTT_CQADupstackTexRetrieval",
    #"PTT_CQADupstackUnixRetrieval",
    #"PTT_CQADupstackWebmastersRetrieval",
    #"PTT_CQADupstackWordpressRetrieval",
    "PTT_DBPedia",
    "PTT_FEVER",
    "PTT_FiQA2018",
    "PTT_HotpotQA",
    "PTT_MSMARCO",
    "PTT_NFCorpus",
    "PTT_NQ",
    "PTT_QuoraRetrieval",
    "PTT_SCIDOCS",
    "PTT_SciFact",
    "PTT_Touche2020",
    "PTT_TRECCOVID",
]

TASK_LIST_TRANSLATED_STS = [
    "PTT_BIOSSES",
    "PTT_SICK-R",
    "PTT_STS12",
    "PTT_STS13",
    "PTT_STS14",
    "PTT_STS15",
    "PTT_STS16",
    "PTT_STS17",
    "PTT_STS22",
    "PTT_STSBenchmark",
    "PTT_SummEval",
]

TASK_LIST_TRANSLATED_RETRIEVAL_LAW = [
    #"PTT_LegalSummarization",
    "PTT_LegalBenchConsumerContractsQA",
    "PTT_LegalBenchCorporateLobbying",
    "PTT_AILACasedocs",
    "PTT_AILAStatutes",
    #"PTT_LeCaRDv2",
    #"PTT_LegalQuAD",
    #"PTT_GerDaLIRSmall",
]

TASK_LIST_TRANSLATED_JINAAI = [
    "PTT_BigPatentClustering",
    "PTT_WikiCitiesClustering",
    "PTT_NarrativeQA"
]

TASK_LIST_TRANSLATED_LEMB = [
    "PTT_LEMBNarrativeQARetrieval",
    "PTT_LEMBNeedleRetrieval",
    "PTT_LEMBPasskeyRetrieval",
    "PTT_LEMBQMSumRetrieval",
    "PTT_LEMBSummScreenFDRetrieval",
    "PTT_LEMBWikimQARetrieval"
]

####################################################

TASK_LIST_TRANSLATED_SMALL_CLASSIFICATION = [
    "PTT_Banking77Classification",
]

TASK_LIST_TRANSLATED_SMALL_CLUSTERING = [
    "PTT_MedrxivClusteringS2S",
    "PTT_TwentyNewsgroupsClustering",
]

TASK_LIST_TRANSLATED_SMALL_PAIR_CLASSIFICATION = [
    "PTT_SprintDuplicateQuestions",
    "PTT_TwitterSemEval2015",
]

TASK_LIST_TRANSLATED_SMALL_RERANKING = [
    "PTT_AskUbuntuDupQuestions",
    "PTT_StackOverflowDupQuestions",
]

TASK_LIST_TRANSLATED_SMALL_RETRIEVAL = [
    "PTT_CQADupstackStatsRetrieval",
    #"PTT_SciFact"
]

TASK_LIST_TRANSLATED_SMALL_STS = [
    "PTT_BIOSSES",
    "PTT_SICK-R",
    "PTT_STS15",
    "PTT_STS16"
]

####################################################

TASK_LIST_EXTERNAL_BITEXT_MINING = [
    #"Tatoeba",
    #"BibleNLPBitextMining",
    "NTREXBitextMining",
    "FloresBitextMining"
]

langs = ["eng_Latn", "spa_Latn", "fra_Latn", "ita_Latn", "deu_Latn", "jpn_Jpan", "kor_Hang", "rus_Cyrl", "arb_Arab", "zho_Hant", "zho_Hans", "pol_Latn", "swe_Latn"]
#["ben_Beng", "swa_Latn", "tur_Latn", "hin_Deva"]

bitext_langs = ["por-eng"]
for lang in langs:
    bitext_langs.append(f"{lang}-por_Latn")

#for lang in langs + ["zho_Hant", "swa_Latn"]:
#    TASK_LIST_EXTERNAL_BITEXT_MINING.append(f"NTREXBitextMining ({lang}-por_Latn)")

#for lang in langs + ["zho_Hans"]:
#    TASK_LIST_EXTERNAL_BITEXT_MINING.append(f"FloresBitextMining ({lang}-por_Latn)")


TASK_LIST_EXTERNAL_CLASSIFICATION = [
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultiHateClassification",
]

TASK_LIST_EXTERNAL_CLUSTERING = [
]

TASK_LIST_EXTERNAL_PAIR_CLASSIFICATION = [
]

TASK_LIST_EXTERNAL_RERANKING = [
]

TASK_LIST_EXTERNAL_RETRIEVAL = [
    "MintakaRetrieval",
    "MultiLongDocRetrieval",
    "XPQARetrieval"
]

TASK_LIST_EXTERNAL_STS = [
    "STSBenchmarkMultilingualSTS"
]

####################################################

TASK_LIST_BITEXT_MINING = [
]

TASK_LIST_CLASSIFICATION = [
    "brazillian_court_decisions_judgment",
    "datalawyer-frases-v0_4",
    "FactNewsFactuality",
    "FactNewsBias",
    "HateBR",
    "PortugueseHateSpeech",
    "Puntuguese",
    "RRIP",
    "ToLD-Br"
]

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = [
    "ASSIN2-RTE",
    "ASSIN-paraphrase",
]

TASK_LIST_RERANKING = [
    "MilkQA-Reranking"
]

TASK_LIST_RETRIEVAL = [
    "Pira2Retrieval"
]

TASK_LIST_STS = [
    "ASSIN2-STS",
    "ASSIN-STS",
    "SICK-BR-STS"
]

TASK_LIST = (
    #TASK_LIST_TRANSLATED_SMALL_CLASSIFICATION
    TASK_LIST_TRANSLATED_SMALL_CLUSTERING
    + TASK_LIST_TRANSLATED_SMALL_PAIR_CLASSIFICATION
    + TASK_LIST_TRANSLATED_SMALL_RERANKING
    + TASK_LIST_TRANSLATED_SMALL_RETRIEVAL
    #+ TASK_LIST_TRANSLATED_SMALL_STS
    + TASK_LIST_EXTERNAL_BITEXT_MINING
    + TASK_LIST_EXTERNAL_CLASSIFICATION
    + TASK_LIST_EXTERNAL_CLUSTERING
    + TASK_LIST_EXTERNAL_PAIR_CLASSIFICATION
    + TASK_LIST_EXTERNAL_RERANKING
    + TASK_LIST_EXTERNAL_RETRIEVAL
    + TASK_LIST_EXTERNAL_STS
    #+ TASK_LIST_BITEXT_MINING
    + TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
    + TASK_LIST_TRANSLATED_RETRIEVAL_LAW
)

import os

FAILED_MODELS = []
MODEL_DONE = []
device_str = "cuda"
#torch.cuda.set_per_process_memory_fraction(0.5, 0)

print("TASK LIST", len(TASK_LIST), TASK_LIST)

def get_next_model():
    global FAILED_MODELS
    with open('embeddings_order.txt', 'r') as f:
        for model in f:
            model = model.strip()
            if not model or model.startswith('#') or model.startswith('//'):
                continue
            if model in FAILED_MODELS or model in MODEL_DONE:
                continue
            if os.path.exists(f"results/{model}") and len(os.listdir(f"results/{model}")) == len(TASK_LIST):
                continue
            return model
    return None
                

model_name = get_next_model()

while model_name is not None:
    model_name = get_next_model()
    logger.info(f"Running task for: {model_name}")
    model = None
    batch_size = 128
    try:
        model = SentenceTransformer(model_name, device=device_str, trust_remote_code=True)
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Failed to load model {model_name}")
        FAILED_MODELS.append(model_name)
        continue

    print("model max seq len", model.max_seq_length)

    model_fail_count = 0
    for task in TASK_LIST:
        retry = True
        while retry:
            retry = False
            try:
                logger.info(f"Running task: {task}")
                evaluation = MTEB(
                    tasks=[task], task_langs=["por-Latn", "por_Latn", "pt", "por"] + bitext_langs
                )  # Remove "en" for running all languages
                evaluation.run(model, output_folder=f"results/{model_name}", batch_size=batch_size)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(traceback.format_exc())
                logger.error("Reducing batch_size")
                if batch_size > 1:
                    retry = True
                    batch_size = batch_size // 2  
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"Failed running of task {task} for the model {model_name}")
                model_fail_count += 1
        if model_fail_count > 5:
            FAILED_MODELS.append(model_name)
            break
        if batch_size < 4:
            batch_size = 4

    MODEL_DONE.append(model_name)

    try:
        del model
        gc.collect()
        if device_str == 'cuda':
            torch.cuda.empty_cache()
        else:
            with torch.cuda.device(device_str):
                torch.cuda.empty_cache()
    except:
        pass