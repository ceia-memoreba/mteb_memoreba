import datasets
from datasets import ClassLabel
from typing import Dict, List, Optional, Union, Callable
import json
import textwrap

import xml.etree.ElementTree as ET
import pandas as pd

logger = datasets.logging.get_logger(__name__)

# Extracted from: 
# - https://huggingface.co/datasets/lener_br
# - https://github.com/peluz/lener-br
# - https://teodecampos.github.io/LeNER-Br/
_LENERBR_KWARGS = dict(
    name = "LeNER-Br",
    description=textwrap.dedent(
            """\
        LeNER-Br is a Portuguese language dataset for named entity recognition applied to legal documents. 
        LeNER-Br consists entirely of manually annotated legislation and legal cases texts and contains tags 
        for persons, locations, time entities, organizations, legislation and legal cases. To compose the dataset, 
        66 legal documents from several Brazilian Courts were collected. Courts of superior and state levels were considered, 
        such as Supremo Tribunal Federal, Superior Tribunal de Justiça, Tribunal de Justiça de Minas Gerais and Tribunal de Contas da União. 
        In addition, four legislation documents were collected, such as "Lei Maria da Penha", giving a total of 70 documents."""
    ),
    task_type="ner",
    label_classes=["ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA"],
    data_urls={
        "train": "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/train/train.conll",
        "validation": "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/dev/dev.conll",
        "test": "https://raw.githubusercontent.com/peluz/lener-br/master/leNER-Br/test/test.conll",
    },  
    citation=textwrap.dedent(
            """\
        @InProceedings{luz_etal_propor2018,
            author = {Pedro H. {Luz de Araujo} and Te\'{o}filo E. {de Campos} and
                    Renato R. R. {de Oliveira} and Matheus Stauffer and
                    Samuel Couto and Paulo Bermejo},
            title = {{LeNER-Br}: a Dataset for Named Entity Recognition in {Brazilian} Legal Text},
            booktitle = {International Conference on the Computational Processing of Portuguese ({PROPOR})},
            publisher = {Springer},
            series = {Lecture Notes on Computer Science ({LNCS})},
            pages = {313--323},
            year = {2018},
            month = {September 24-26},
            address = {Canela, RS, Brazil},	  
            doi = {10.1007/978-3-319-99722-3_32},
            url = {https://teodecampos.github.io/LeNER-Br/},
        }"""
    ),
    url="https://teodecampos.github.io/LeNER-Br/",
)

# Extracted from: 
# - https://huggingface.co/datasets/assin2
# - https://sites.google.com/view/assin2
# - https://github.com/ruanchaves/assin
_ASSIN2_BASE_KWARGS = dict(
    description=textwrap.dedent(
            """\
        The ASSIN 2 corpus is composed of rather simple sentences. Following the procedures of SemEval 2014 Task 1.
        The training and validation data are composed, respectively, of 6,500 and 500 sentence pairs in Brazilian Portuguese,
        annotated for entailment and semantic similarity. Semantic similarity values range from 1 to 5, and text entailment
        classes are either entailment or none. The test data are composed of approximately 3,000 sentence pairs with the same
        annotation. All data were manually annotated."""
    ),
    data_urls={
        "train": "https://github.com/ruanchaves/assin/raw/master/sources/assin2-train-only.xml",
        "validation": "https://github.com/ruanchaves/assin/raw/master/sources/assin2-dev.xml",
        "test": "https://github.com/ruanchaves/assin/raw/master/sources/assin2-test.xml",
    },  
    citation=textwrap.dedent(
            """\
        @inproceedings{real2020assin,
            title={The assin 2 shared task: a quick overview},
            author={Real, Livy and Fonseca, Erick and Oliveira, Hugo Goncalo},
            booktitle={International Conference on Computational Processing of the Portuguese Language},
            pages={406--412},
            year={2020},
            organization={Springer}
        }"""
    ),
    url="https://sites.google.com/view/assin2",
)
_ASSIN2_RTE_KWARGS = dict(
    name = "assin2-rte",
    task_type="rte",
    label_classes=["NONE", "ENTAILMENT"],
    **_ASSIN2_BASE_KWARGS
)
_ASSIN2_STS_KWARGS = dict(
    name = "assin2-sts",
    task_type="sts",
    **_ASSIN2_BASE_KWARGS
)

# Extracted from: 
# - https://huggingface.co/datasets/ruanchaves/hatebr
# - https://github.com/franciellevargas/HateBR
_HATEBR_META_KWARGS = dict(
    description=textwrap.dedent(
            """\
        HateBR is the first large-scale expert annotated dataset of Brazilian Instagram comments for abusive language detection 
        on the web and social media. The HateBR was collected from Brazilian Instagram comments of politicians and manually annotated 
        by specialists. It is composed of 7,000 documents annotated according to three different layers: a binary classification (offensive 
        versus non-offensive comments), offensiveness-level (highly, moderately, and slightly offensive messages), and nine hate speech 
        groups (xenophobia, racism, homophobia, sexism, religious intolerance, partyism, apology for the dictatorship, antisemitism, 
        and fatphobia). Each comment was annotated by three different annotators and achieved high inter-annotator agreement. Furthermore, 
        baseline experiments were implemented reaching 85% of F1-score outperforming the current literature dataset baselines for 
        the Portuguese language. We hope that the proposed expert annotated dataset may foster research on hate speech detection in the 
        Natural Language Processing area."""
    ),
    task_type="classification",
    file_type="csv",
    data_urls={
        "train": "https://raw.githubusercontent.com/franciellevargas/HateBR/2d18c5b9410c2dfdd6d5394caa54d608857dae7c/dataset/HateBR.csv"
    },
    citation=textwrap.dedent(
            """\
        @inproceedings{vargas2022hatebr,
            title={HateBR: A Large Expert Annotated Corpus of Brazilian Instagram Comments for Offensive Language and Hate Speech Detection},
            author={Vargas, Francielle and Carvalho, Isabelle and de G{\'o}es, Fabiana Rodrigues and Pardo, Thiago and Benevenuto, Fabr{\'\i}cio},
            booktitle={Proceedings of the Thirteenth Language Resources and Evaluation Conference},
            pages={7174--7183},
            year={2022}
        }"""
    ),
    url="https://github.com/franciellevargas/HateBR",
    indexes_url="https://huggingface.co/datasets/ruanchaves/hatebr/raw/main/indexes.json"
)
hatebr_level_map = {
    "0": "non-offensive",
    "1": "slightly",
    "2": "moderately",
    "3": "highly",
}
_HATEBR_LEVEL_KWARGS = dict(
    name = "HateBR_offensive_level",
    text_and_label_columns=["instagram_comments", "offensiveness_levels"],
    label_classes=["non-offensive", "slightly", "moderately", "highly"],
    process_label = lambda x: hatebr_level_map[x],
    **_HATEBR_META_KWARGS
)
hatebr_binary_map = {
    "0": "non-offensive",
    "1": "offensive",
}
_HATEBR_BINARY_KWARGS = dict(
    name = "HateBR_offensive_binary",
    text_and_label_columns=["instagram_comments", "offensive_language"],
    label_classes=["non-offensive", "offensive"],
    process_label = lambda x: hatebr_binary_map[x],
    **_HATEBR_META_KWARGS
)



# Extracted from: 
# - https://github.com/ulysses-camara/ulysses-ner-br

_ULYSSESNER_META_KWARGS = dict(
    description=textwrap.dedent(
            """\
        UlyssesNER-Br is a corpus of Brazilian Legislative Documents for NER with quality baselines. 
        The presented corpus consists of bills and legislative consultations from Brazilian Chamber of Deputies.
        UlyssesNER-Br has seven semantic classes or categories. Based on HAREM,
        we defined five typical categories: person, location, organization, event and date.
        In addition, we defined two specific semantic classes for the legislative domain:
        law foundation and law product. The law foundation category makes reference to
        entities related to laws, resolutions, decrees, as well as to domain-specific entities
        such as bills, which are law proposals being discussed by the parliament, and legislative consultations, 
        also known as job requests made by the parliamentarians.
        The law product entity refers to systems, programs, and other products created
        from legislation."""
    ),
    task_type="ner",
    citation=textwrap.dedent(
            """\
        @InProceedings{10.1007/978-3-030-98305-5_1,
            author="Albuquerque, Hidelberg O.
            and Costa, Rosimeire
            and Silvestre, Gabriel
            and Souza, Ellen
            and da Silva, N{\'a}dia F. F.
            and Vit{\'o}rio, Douglas
            and Moriyama, Gyovana
            and Martins, Lucas
            and Soezima, Luiza
            and Nunes, Augusto
            and Siqueira, Felipe
            and Tarrega, Jo{\~a}o P.
            and Beinotti, Joao V.
            and Dias, Marcio
            and Silva, Matheus
            and Gardini, Miguel
            and Silva, Vinicius
            and de Carvalho, Andr{\'e} C. P. L. F.
            and Oliveira, Adriano L. I.",
            editor="Pinheiro, Vl{\'a}dia
            and Gamallo, Pablo
            and Amaro, Raquel
            and Scarton, Carolina
            and Batista, Fernando
            and Silva, Diego
            and Magro, Catarina
            and Pinto, Hugo",
            title="UlyssesNER-Br: A Corpus of Brazilian Legislative Documents for Named Entity Recognition",
            booktitle="Computational Processing of the Portuguese Language",
            year="2022",
            publisher="Springer International Publishing",
            address="Cham",
            pages="3--14",
            isbn="978-3-030-98305-5"
        }
        @InProceedings{10.1007/978-3-031-16474-3_62,
            author="Costa, Rosimeire
            and Albuquerque, Hidelberg Oliveira
            and Silvestre, Gabriel
            and Silva, N{\'a}dia F{\'e}lix F.
            and Souza, Ellen
            and Vit{\'o}rio, Douglas
            and Nunes, Augusto
            and Siqueira, Felipe
            and Pedro Tarrega, Jo{\~a}o
            and Vitor Beinotti, Jo{\~a}o
            and de Souza Dias, M{\'a}rcio
            and Pereira, Fab{\'i}ola S. F.
            and Silva, Matheus
            and Gardini, Miguel
            and Silva, Vinicius
            and de Carvalho, Andr{\'e} C. P. L. F.
            and Oliveira, Adriano L. I.",
            editor="Marreiros, Goreti
            and Martins, Bruno
            and Paiva, Ana
            and Ribeiro, Bernardete
            and Sardinha, Alberto",
            title="Expanding UlyssesNER-Br Named Entity Recognition Corpus with Informal User-Generated Text",
            booktitle="Progress in Artificial Intelligence",
            year="2022",
            publisher="Springer International Publishing",
            address="Cham",
            pages="767--779",
            isbn="978-3-031-16474-3"
        }"""
    ),
    url="https://github.com/ulysses-camara/ulysses-ner-br",
)
_ULYSSESNER_PL_KWARGS = dict(
    name = "UlyssesNER-Br-PL-coarse",
    data_urls = {
        "train": "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/PL_corpus_conll/pl_corpus_categorias/train.txt",
        "validation":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/PL_corpus_conll/pl_corpus_categorias/valid.txt",
        "test":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/PL_corpus_conll/pl_corpus_categorias/test.txt",
    },
    label_classes = ['DATA', 'EVENTO', 'FUNDAMENTO', 'LOCAL', 'ORGANIZACAO', 'PESSOA', 'PRODUTODELEI'],
    **_ULYSSESNER_META_KWARGS
)
_ULYSSESNER_C_KWARGS = dict(
    name = "UlyssesNER-Br-C-coarse",
    data_urls = {
        "train": "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/C_corpus_conll/c_corpus_categorias/train.txt",
        "validation":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/C_corpus_conll/c_corpus_categorias/valid.txt",
        "test":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/C_corpus_conll/c_corpus_categorias/test.txt",
    },
    label_classes = ['DATA', 'EVENTO', 'FUNDAMENTO', 'LOCAL', 'ORGANIZACAO', 'PESSOA', 'PRODUTODELEI'],
    **_ULYSSESNER_META_KWARGS
)

_ULYSSESNER_PL_TIPOS_KWARGS = dict(
    name = "UlyssesNER-Br-PL-fine",
    data_urls = {
        "train": "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/train.txt",
        "validation":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/valid.txt",
        "test":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/PL_corpus_conll/pl_corpus_tipos/test.txt",
    },
    label_classes = ['DATA', 'EVENTO', 'FUNDapelido', 'FUNDlei', 'FUNDprojetodelei', 'LOCALconcreto', 'LOCALvirtual', \
                    'ORGgovernamental', 'ORGnaogovernamental', 'ORGpartido', 'PESSOAcargo', 'PESSOAgrupocargo', 'PESSOAindividual', \
                    'PRODUTOoutros', 'PRODUTOprograma', 'PRODUTOsistema'],
    **_ULYSSESNER_META_KWARGS
)
_ULYSSESNER_C_TIPOS_KWARGS = dict(
    name = "UlyssesNER-Br-C-fine",
    data_urls = {
        "train": "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/C_corpus_conll/c_corpus_tipos/train.txt",
        "validation":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/C_corpus_conll/c_corpus_tipos/valid.txt",
        "test":  "https://github.com/ulysses-camara/ulysses-ner-br/raw/main/annotated-corpora/C_corpus_conll/c_corpus_tipos/test.txt",
    },
    label_classes = ['DATA', 'EVENTO', 'FUNDapelido', 'FUNDlei', 'FUNDprojetodelei', 'LOCALconcreto', 'LOCALvirtual', \
                    'ORGgovernamental', 'ORGnaogovernamental', 'ORGpartido', 'PESSOAcargo', 'PESSOAgrupocargo', 'PESSOAgrupoind', \
                    'PESSOAindividual', 'PRODUTOoutros', 'PRODUTOprograma', 'PRODUTOsistema'],
    **_ULYSSESNER_META_KWARGS
)

_BRAZILIAN_COURT_DECISIONS_JUDGMENT = dict(
    name = "brazilian_court_decisions_judgment",
    task_type = "classification",
    data_urls = "joelito/brazilian_court_decisions",
    text_and_label_columns = ["decision_description", "judgment_label"],
    file_type="hf_dataset",
    url = "https://github.com/lagefreitas/predicting-brazilian-court-decisions",
    description =textwrap.dedent(
            """\
        The dataset is a collection of 4043 Ementa (summary) court decisions and their metadata from the Tribunal de 
        Justiça de Alagoas (TJAL, the State Supreme Court of Alagoas (Brazil). The court decisions are labeled according 
        to 7 categories and whether the decisions were unanimous on the part of the judges or not. The dataset 
        supports the task of Legal Judgment Prediction."""
    ),
    citation = textwrap.dedent(
        """\
        @article{Lage-Freitas2022,
          author = {Lage-Freitas, Andr{\'{e}} and Allende-Cid, H{\'{e}}ctor and Santana, Orivaldo and Oliveira-Lage, L{\'{i}}via},
          doi = {10.7717/peerj-cs.904},
          issn = {2376-5992},
          journal = {PeerJ. Computer science},
          keywords = {Artificial intelligence,Jurimetrics,Law,Legal,Legal NLP,Legal informatics,Legal outcome forecast,Litigation prediction,Machine learning,NLP,Portuguese,Predictive algorithms,judgement prediction},
          language = {eng},
          month = {mar},
          pages = {e904--e904},
          publisher = {PeerJ Inc.},
          title = {{Predicting Brazilian Court Decisions}},
          url = {https://pubmed.ncbi.nlm.nih.gov/35494851 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044329/},
          volume = {8},
          year = {2022}
        }"""
    ),
    label_classes = ["no", "partial", "yes"]
)
_BRAZILIAN_COURT_DECISIONS_UNANIMITY = {
    **_BRAZILIAN_COURT_DECISIONS_JUDGMENT,
    "name": "brazilian_court_decisions_unanimity",
    "text_and_label_columns": ["decision_description", "unanimity_label"],
    "label_classes": ["unanimity", "not-unanimity"],
}
HAREM_BASE_KWARGS = dict(
    description=textwrap.dedent(
            """\
        The HAREM is a Portuguese language corpus commonly used for Named Entity Recognition tasks. It includes about 93k words, from 129 different texts,
        from several genres, and language varieties. The split of this dataset version follows the division made by [1], where 7% HAREM
        documents are the validation set and the miniHAREM corpus (with about 65k words) is the test set. There are two versions of the dataset set,
        a version that has a total of 10 different named entity classes (Person, Organization, Location, Value, Date, Title, Thing, Event,
        Abstraction, and Other) and a "selective" version with only 5 classes (Person, Organization, Location, Value, and Date).
        It's important to note that the original version of the HAREM dataset has 2 levels of NER details, namely "Category" and "Sub-type".
        The dataset version processed here ONLY USE the "Category" level of the original dataset.
        [1] Souza, Fábio, Rodrigo Nogueira, and Roberto Lotufo. "BERTimbau: Pretrained BERT Models for Brazilian Portuguese."
        Brazilian Conference on Intelligent Systems. Springer, Cham, 2020."""
    ),
    task_type="ner",
    data_urls="harem",
    file_type="hf_dataset",
    text_and_label_columns = ["tokens", "ner_tags"],
    citation=textwrap.dedent(
            """\
        @inproceedings{santos2006harem,
            title={Harem: An advanced ner evaluation contest for portuguese},
            author={Santos, Diana and Seco, Nuno and Cardoso, Nuno and Vilela, Rui},
            booktitle={quot; In Nicoletta Calzolari; Khalid Choukri; Aldo Gangemi; Bente Maegaard; Joseph Mariani; Jan Odjik; Daniel Tapias (ed) Proceedings of the 5 th International Conference on Language Resources and Evaluation (LREC'2006)(Genoa Italy 22-28 May 2006)},
            year={2006}
        }"""
    ),
    url="https://www.linguateca.pt/primeiroHAREM/harem_coleccaodourada_en.html",
)
HAREM_DEFAULT_KWARGS = dict(
    name = "harem-default",
    extra_configs = {"name": "default"},
    label_classes = ["PESSOA", "ORGANIZACAO", "LOCAL", "TEMPO", "VALOR", "ABSTRACCAO", "ACONTECIMENTO", "COISA", "OBRA", "OUTRO"],
    **HAREM_BASE_KWARGS
)
HAREM_SELECTIVE_KWARGS = dict(
    name = "harem-selective",
    extra_configs = {"name": "selective"},
    label_classes = ["PESSOA", "ORGANIZACAO", "LOCAL", "TEMPO", "VALOR"],
    **HAREM_BASE_KWARGS
)

_MAPA_BASE_KWARGS = dict(
    task_type = "ner",
    data_urls = "joelito/mapa",
    file_type="hf_dataset",
    url = "",
    description =textwrap.dedent(
            """\
        The dataset consists of 12 documents (9 for Spanish due to parsing errors) taken from EUR-Lex, 
        a multilingual corpus of court decisions and legal dispositions in the 24 official languages 
        of the European Union. The documents have been annotated for named entities following the 
        guidelines of the MAPA project which foresees two annotation level, a general and a more 
        fine-grained one. The annotated corpus can be used for named entity recognition/classification."""
    ),
    citation = textwrap.dedent(
        """\
        @article{DeGibertBonet2022,
            author = {{de Gibert Bonet}, Ona and {Garc{\'{i}}a Pablos}, Aitor and Cuadros, Montse and Melero, Maite},
            journal = {Proceedings of the Language Resources and Evaluation Conference},
            number = {June},
            pages = {3751--3760},
            title = {{Spanish Datasets for Sensitive Entity Detection in the Legal Domain}},
            url = {https://aclanthology.org/2022.lrec-1.400},
            year = {2022}
        }"""
    )
)
_MAPA_BASE_KWARGS['filter'] = lambda item: item["language"] == "pt"
_MAPA_COARSE_KWARGS = dict(
    name = "mapa_pt_coarse",
    text_and_label_columns = ["tokens", "coarse_grained"],
    label_classes = ['ADDRESS', 'AMOUNT', 'DATE', 'ORGANISATION', 'PERSON', 'TIME'],
    **_MAPA_BASE_KWARGS
)

_MAPA_FINE_KWARGS = dict(
    name = "mapa_pt_fine",
    text_and_label_columns = ["tokens", "fine_grained"],
    label_classes = ['AGE', 'BUILDING', 'CITY', 'COUNTRY', 'DAY', 'ETHNIC CATEGORY', 
                     'FAMILY NAME', 'INITIAL NAME', 'MARITAL STATUS', 'MONTH', 'NATIONALITY', 
                     'PLACE', 'PROFESSION', 'ROLE', 'STANDARD ABBREVIATION', 'TERRITORY', 
                     'TITLE', 'TYPE', 'UNIT', 'URL', 'VALUE', 'YEAR'],
    **_MAPA_BASE_KWARGS
)
    

_MULTIEURLEX_BASE_KWARGS = dict(
    name = "multi_eurlex_pt",
    task_type = "multilabel_classification",
    data_urls = "multi_eurlex",
    file_type="hf_dataset",
    extra_configs = {"language": "pt", "label_level": "level_1"},
    text_and_label_columns = ["text", "labels"],
    url = "https://github.com/nlpaueb/MultiEURLEX/",
    description =textwrap.dedent(
            """\
         MultiEURLEX comprises 65k EU laws in 23 official EU languages. 
         Each EU law has been annotated with EUROVOC concepts (labels) by the Publication Office of EU. 
         Each EUROVOC label ID is associated with a label descriptor, e.g., [60, agri-foodstuffs], 
         [6006, plant product], [1115, fruit]. The descriptors are also available in the 23 languages. 
         Chalkidis et al. (2019) published a monolingual (English) version of this dataset, called EUR-LEX, 
         comprising 57k EU laws with the originally assigned gold labels."""
    ),
    citation = textwrap.dedent(
        """\
        @InProceedings{chalkidis-etal-2021-multieurlex,
          author = {Chalkidis, Ilias  
                        and Fergadiotis, Manos
                        and Androutsopoulos, Ion},
          title = {MultiEURLEX -- A multi-lingual and multi-label legal document 
                       classification dataset for zero-shot cross-lingual transfer},
          booktitle = {Proceedings of the 2021 Conference on Empirical Methods
                       in Natural Language Processing},
          year = {2021},
          publisher = {Association for Computational Linguistics},
          location = {Punta Cana, Dominican Republic},
          url = {https://arxiv.org/abs/2109.00904}
        }"""
    ),
    label_classes = [
        "100149","100160","100148","100147","100152","100143","100156",
        "100158","100154","100153","100142","100145","100150","100162",
        "100159","100144","100151","100157","100161","100146","100155"
    ]
)

# Extracted from: 
# - https://huggingface.co/datasets/ruanchaves/hatebr
# - https://github.com/franciellevargas/HateBR
_PORTUGUESE_HATE_SPEECH_META_KWARGS = dict(
    description=textwrap.dedent(
            """\
        The dataset is composed of 5,668 tweets. For its annotation, we defined two different schemes used by
        annotators with different levels of expertise. Firstly, non-experts annotated the tweets with binary
        labels (‘hate’ vs. ‘no-hate’). Secondly, expert annotators classified the tweets following a fine-grained
        hierarchical multiple label scheme with 81 hate speech categories in total. The inter-annotator agreement
        varied from category to category, which reflects the insight that some types of hate speech are more subtle
        than others and that their detection depends on personal perception. This hierarchical annotation scheme is
        the main contribution of the presented work, as it facilitates the identification of different types of
        hate speech and their intersections.
        """
    ),
    task_type="classification",
    file_type="csv",
    citation=textwrap.dedent(
            """\
        @inproceedings{fortuna-etal-2019-hierarchically,
            title = "A Hierarchically-Labeled {P}ortuguese Hate Speech Dataset",
            author = "Fortuna, Paula  and
            Rocha da Silva, Jo{\~a}o  and
            Soler-Company, Juan  and
            Wanner, Leo  and
            Nunes, S{\'e}rgio",
            editor = "Roberts, Sarah T.  and
            Tetreault, Joel  and
            Prabhakaran, Vinodkumar  and
            Waseem, Zeerak",
            booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
            month = aug,
            year = "2019",
            address = "Florence, Italy",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/W19-3510",
            doi = "10.18653/v1/W19-3510",
            pages = "94--104"
        }"""
    ),
    url="https://github.com/paulafortuna/Portuguese-Hate-Speech-Dataset"
)
portuguese_hate_binary_map = {
    "0": "no-hate",
    "1": "hate",
}
_PORTUGUESE_HATE_SPEECH_BINARY_KWARGS = dict(
    name = "Portuguese_Hate_Speech_binary",
    data_urls={
        "train": "https://raw.githubusercontent.com/paulafortuna/Portuguese-Hate-Speech-Dataset/master/2019-05-28_portuguese_hate_speech_binary_classification.csv"
    },
    text_and_label_columns=["text", "hatespeech_comb"],
    label_classes=["no-hate", "hate"],
    process_label = lambda x: portuguese_hate_binary_map[x],
    indexes_url="metadata/portuguese_hate_speech_binary_indexes.json",
    **_PORTUGUESE_HATE_SPEECH_META_KWARGS
)

class PTBenchmarkConfig(datasets.BuilderConfig):
    """BuilderConfig for PTBenchmark."""

    def __init__(
        self,
        task_type: str,
        data_urls: Union[str, Dict[str, str]],
        citation: str,
        url: str,
        label_classes: Optional[List[Union[str, int]]] = None,
        file_type: Optional[str] = None, #filetype (csv, tsc, jsonl)
        text_and_label_columns: Optional[List[str]] = None, #columns for train, dev and test for csv datasets
        indexes_url: Optional[str] = None, #indexes for train, dev and test for single file datasets
        process_label: Callable[[str], str] = lambda x: x,
        filter: Callable = lambda x: True,
        extra_configs: Dict = {},
        **kwargs,
    ):
        """BuilderConfig for GLUE.
        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          label_column: `string`, name of the column in the tsv file corresponding
            to the label
          data_url: `string`, url to download the zip file from
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(PTBenchmarkConfig, self).__init__(version=datasets.Version("1.0.3", ""), **kwargs)
        self.label_classes = label_classes
        self.task_type = task_type
        self.data_urls = data_urls
        self.citation = citation
        self.url = url
        self.file_type = file_type
        self.text_and_label_columns = text_and_label_columns
        self.indexes_url = indexes_url
        self.process_label = process_label
        self.filter = filter
        self.extra_configs = extra_configs

def _get_classification_features(config: PTBenchmarkConfig):
    return datasets.Features(
        {
            "idx": datasets.Value("int32"),
            "sentence": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=config.label_classes),
        }
    )

def _get_multilabel_classification_features(config: PTBenchmarkConfig):
    return datasets.Features(
        {
            "idx": datasets.Value("int32"),
            "sentence": datasets.Value("string"),
            "labels": datasets.Sequence(
                datasets.features.ClassLabel(names=config.label_classes)
            ),
        }
    )

def _get_ner_features(config: PTBenchmarkConfig):
    bio_labels = ["O"]
    for label_name in config.label_classes:
        bio_labels.append("B-" + label_name)
        bio_labels.append("I-" + label_name)
    return datasets.Features(
        {
            "idx": datasets.Value("int32"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "ner_tags": datasets.Sequence(
                datasets.features.ClassLabel(names=bio_labels)
            ),
        }
    )

def _get_rte_features(config: PTBenchmarkConfig):
    return datasets.Features(
        {
            "idx": datasets.Value("int32"),
            "sentence1": datasets.Value("string"),
            "sentence2": datasets.Value("string"),
            "label": datasets.features.ClassLabel(names=config.label_classes),
        }
    )

def _get_sts_features(config: PTBenchmarkConfig = None):
    return datasets.Features(
        {
            "idx": datasets.Value("int32"),
            "sentence1": datasets.Value("string"),
            "sentence2": datasets.Value("string"),
            "label": datasets.Value("float32"),
        }
    )

def _csv_generator(file_path: str,
                   config: PTBenchmarkConfig,
                   indexes_path: Optional[str] = None,
                   split: Optional[str] = None
                   ):
    """Yields examples."""
    df = pd.read_csv(file_path)
    columns = config.text_and_label_columns
    df = df[columns]

    with open(indexes_path, "r") as f:
        indexes= json.load(f)
    split_indexes = indexes[split]
    df_split = df.iloc[split_indexes]

    for id_, row in df_split.iterrows():
        example = {
            "idx": id_,
            "sentence": str(row[columns[0]]),
            "label": config.process_label(str(row[columns[-1]]))
        }
        yield id_, example

def _conll_ner_generator(file_path: str, config: PTBenchmarkConfig):
    with open(file_path, encoding="utf-8") as f:

        guid = 0
        tokens = []
        ner_tags = []

        for line in f:
            if line == "" or line == "\n":
                if tokens:
                    # Filter for Ulysses empty data
                    if len(tokens) == 1 and tokens[0] == '.':
                        guid += 1
                        tokens = []
                        ner_tags = []
                        continue
                    yield guid, {
                        "idx": guid,
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    guid += 1
                    tokens = []
                    ner_tags = []
            else:
                splits = line.split(" ")
                tokens.append(splits[0])
                ner_tags.append(config.process_label(splits[1].rstrip()))

        # last example
        yield guid, {
            "idx": guid,
            "tokens": tokens,
            "ner_tags": ner_tags,
        }

def _assin2_generator(file_path, config: PTBenchmarkConfig):
    """Yields examples."""
    id_ = 0

    with open(file_path, "rb") as f:

        tree = ET.parse(f)
        root = tree.getroot()

        task_type = config.task_type

        for pair in root:

            example = {
                "idx": int(pair.attrib.get("id")),
                "sentence1": pair.find(".//t").text,
                "sentence2": pair.find(".//h").text
            }
            if task_type == "rte":
                example["label"] = pair.attrib.get("entailment").upper()
            elif task_type == "sts":
                example["label"] = float(config.process_label(pair.attrib.get("similarity")))

            yield id_, example
            id_ += 1

def _hf_dataset_generator(split, config: PTBenchmarkConfig):
    dataset = datasets.load_dataset(config.data_urls, split=split, **config.extra_configs)
    feature_col, label_col = config.text_and_label_columns
    
    target_feature_col, target_label_col = feature_col, label_col
    if config.task_type == "classification":
        target_feature_col, target_label_col = "sentence", "label"
    elif config.task_type == "multilabel_classification":
        target_feature_col, target_label_col = "sentence", "labels"
    elif config.task_type == "ner":
        target_feature_col, target_label_col = "tokens", "ner_tags"

    for id, item in enumerate(dataset):
        #filter invalid items
        if not config.filter(item):
            continue

        label = item[label_col]
        #Convert label to original text
        if isinstance(dataset.features[label_col], ClassLabel):
            if isinstance(label, list):
                label = [dataset.features[label_col].int2str(l) for l in label]
            else:
                label = dataset.features[label_col].int2str(label)
        
        #Process label
        if isinstance(label, list):
            label = [config.process_label(l) for l in label]
        else:
            label = config.process_label(label)

        #Filter out invalid classes
        if config.task_type != "ner":
            if isinstance(label, list):
                invalid = False
                for i in range(len(label)):
                    if label[i] not in config.label_classes:
                        invalid = True
                        break
                if invalid:
                    continue
            else:
                if label not in config.label_classes:
                    continue

        yield id, {
            "idx": id,
            target_feature_col: item[feature_col],
            target_label_col: label,
        }

class PTBenchmark(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PTBenchmarkConfig(
            **CONFIG_KWARGS
        ) \
        for CONFIG_KWARGS in \
            [_LENERBR_KWARGS, _ASSIN2_RTE_KWARGS, _ASSIN2_STS_KWARGS, _HATEBR_BINARY_KWARGS, _HATEBR_LEVEL_KWARGS,
            _ULYSSESNER_PL_KWARGS, _ULYSSESNER_C_KWARGS, _ULYSSESNER_PL_TIPOS_KWARGS,
            _ULYSSESNER_C_TIPOS_KWARGS, _BRAZILIAN_COURT_DECISIONS_JUDGMENT, 
            _BRAZILIAN_COURT_DECISIONS_UNANIMITY, HAREM_DEFAULT_KWARGS, HAREM_SELECTIVE_KWARGS, 
            _MULTIEURLEX_BASE_KWARGS, _MAPA_COARSE_KWARGS, _MAPA_FINE_KWARGS, _PORTUGUESE_HATE_SPEECH_BINARY_KWARGS]
    ]

    def _info(self) -> datasets.DatasetInfo:
        features = None
        if self.config.task_type == "classification":
            features = _get_classification_features(self.config)
        elif self.config.task_type == "multilabel_classification":
            features = _get_multilabel_classification_features(self.config)
        elif self.config.task_type == "ner":
            features = _get_ner_features(self.config)
        elif self.config.task_type == "rte":
            features = _get_rte_features(self.config)
        elif self.config.task_type == "sts":
            features = _get_sts_features(self.config)
        
        return datasets.DatasetInfo(
            description=self.config.description,
            homepage=self.config.url,
            citation=self.config.citation,
            supervised_keys=None,
            features=features
        )
        
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        if self.config.file_type == 'hf_dataset':
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={"split": split},  # These kwargs will be passed to _generate_examples
                )
                for split in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
            ]
        data_urls = self.config.data_urls.copy()
        if self.config.indexes_url is not None:
            data_urls['indexes'] = self.config.indexes_url
        file_paths = dl_manager.download_and_extract(data_urls)

        if self.config.indexes_url is None:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"file_path": file_paths["train"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"file_path": file_paths["validation"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"file_path": file_paths["test"]},
                )
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"file_path": file_paths["train"], "indexes_path": file_paths["indexes"], "split": "train"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"file_path": file_paths["train"], "indexes_path": file_paths["indexes"], "split": "validation"},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"file_path": file_paths["train"], "indexes_path": file_paths["indexes"], "split": "test"},
                )
            ]
    
    def _generate_examples(
        self,
        file_path: Optional[str] = None,
        indexes_path: Optional[str] = None,
        split: Optional[str] = None
    ):
        logger.info("⏳ Generating examples from = %s", file_path)
        if self.config.file_type == "hf_dataset":
            yield from _hf_dataset_generator(split, self.config)
            return

        if self.config.task_type == "classification":
            if self.config.file_type == "csv":
                yield from _csv_generator(
                    file_path, 
                    self.config,
                    indexes_path=indexes_path,
                    split=split
                )
        elif self.config.task_type == "multilabel_classification":
            pass
        elif self.config.task_type == "ner":
            yield from _conll_ner_generator(file_path, self.config)
        elif self.config.task_type == "rte":
            if "assin2" in self.config.name:
                yield from _assin2_generator(file_path, self.config)
        elif self.config.task_type == "sts":
            if "assin2" in self.config.name:
                yield from _assin2_generator(file_path, self.config)
            
