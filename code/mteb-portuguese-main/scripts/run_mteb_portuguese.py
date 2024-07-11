"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from mteb import MTEB

logging.basicConfig(level=logging.INFO)

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
    "PTT_LegalSummarization",
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
    "PTT_SciFact"
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
    + TASK_LIST_TRANSLATED_SMALL_STS
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

model_name = "mteb-pt/average_pt_nilc_word2vec_skip_s50"
model = SentenceTransformer(model_name)

for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    evaluation = MTEB(
        tasks=[task], task_langs=["por-Latn", "por_Latn", "pt", "por"] + bitext_langs
    )  # Remove "en" for running all languages
    evaluation.run(model, output_folder=f"results/{model_name}")
