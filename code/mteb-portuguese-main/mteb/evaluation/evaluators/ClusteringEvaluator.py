from __future__ import annotations

import logging

import numpy as np
import sklearn
import sklearn.cluster

from .Evaluator import Evaluator
from .utils import cos_sim, sha1_hash

logger = logging.getLogger(__name__)


class ClusteringEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        clustering_batch_size=500,
        batch_size=32,
        limit=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        #if limit is not None:
        #    sentences = sentences[:limit]
        #    labels = labels[:limit]
        self.dataset = dataset
        #self.sentences = sentences
        #self.labels = labels
        self.limit = limit
        self.clustering_batch_size = clustering_batch_size
        self.batch_size = batch_size

    def __call__(self, model):
        v_measures = []

        all_corpus_dict = {}
        all_corpus_hashes = []
        samples = []
        for sample in self.dataset:
            sentences = sample["sentences"]
            labels = sample["labels"]
            if self.limit is not None:
                sentences = sentences[:self.limit]
                labels = labels[:self.limit]
            samples.append((sentences, labels))
            for s in sentences:
                hash = sha1_hash(s)
                all_corpus_dict[hash] = s
                all_corpus_hashes.append(hash)
        
        all_unique_docs = []
        all_unique_hashes = []
        for hash, doc in all_corpus_dict.items():
            all_unique_docs.append(doc)
            all_unique_hashes.append(hash)

        logger.info(f"Encoding {len(all_unique_docs)} sentences...")
        unique_corpus_embeddings = np.asarray(
            model.encode(all_unique_docs, batch_size=self.batch_size)
        )

        k = 0
        for sentences, labels in samples:
            sample_length = len(sentences)
            corpus_embeddings = np.zeros((sample_length, unique_corpus_embeddings.shape[1]), dtype=unique_corpus_embeddings.dtype)
            for i, hash in enumerate(all_corpus_hashes[k:k+sample_length]):
                corpus_embeddings[i] = unique_corpus_embeddings[all_unique_hashes.index(hash)]
            k += sample_length

            logger.info("Fitting Mini-Batch K-Means model...")
            clustering_model = sklearn.cluster.MiniBatchKMeans(
                n_clusters=len(set(labels)),
                batch_size=self.clustering_batch_size,
                n_init="auto",
            )
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_

            logger.info("Evaluating...")
            v_measure = sklearn.metrics.cluster.v_measure_score(
                labels, cluster_assignment
            )
            v_measures.append(v_measure)

        return {"v_measures": v_measures}
