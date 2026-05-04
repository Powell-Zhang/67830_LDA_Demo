import numpy as np
from scipy.special import digamma, polygamma
from datasets import load_dataset 
from tqdm import tqdm
def load_wikipedia(num_articles=1000, min_len=200):
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    data = []
    pbar = tqdm(total=num_articles, desc="loading wikipedia articles")
    for example in ds["train"]:
        if len(example["text"].split()) > min_len:
            data.append(example["text"])
            pbar.update(1)
        if len(data) == num_articles:
            break
    pbar.close()
    return data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

import string
from tqdm.contrib.concurrent import process_map

STOPWORDS = set(stopwords.words("english"))


def preprocess_data(data):
    return process_map(
        preprocess_text,
        data,
        max_workers=None,   
        chunksize=10,
        desc="Tokenizing Text"     
    )

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        t for t in tokens
        if t not in string.punctuation
        and t.isalpha()
        and t not in STOPWORDS
    ]
    return tokens

def build_vocab(tokenized_data, min_df = 5, max_df = 0.85):
    doc_freq = Counter()
    for doc in tqdm(tokenized_data, "Building Vocab"):
        doc_freq.update(set(doc))   # set() so each word counts once per doc

    n_docs = len(tokenized_data)
    vocab = {
        word for word, df in doc_freq.items()
        if min_df <= df <= max_df * n_docs      # drop too-rare and too-common
    }
    return vocab, doc_freq

def get_ids(tokenized_data, vocab):
    word_to_id = {word: i for i, word in enumerate(sorted(vocab))}
    corpus_word_ids = []
    for article in tqdm(tokenized_data, desc="Converting words to ids"):
        doc_ids = [word_to_id[token] for token in article if token in word_to_id]
        corpus_word_ids.append(doc_ids)
    return corpus_word_ids, word_to_id
import torch
import torch.nn.functional as F
from scipy.special import digamma
from itertools import groupby

# Check GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _digamma_torch(x):
    """torch.special.digamma works on tensors directly"""
    return torch.special.digamma(x)

def _estep_batch_gpu(unique_ids_batch, counts_batch, log_beta_T, alpha, max_iter=100, tol=1e-3):
    """
    Process a batch of same-length documents on GPU.
    unique_ids_batch: (batch_size, n_unique) int64 tensor
    counts_batch:     (batch_size, n_unique) float64 tensor
    log_beta_T:       (v, k) float32 tensor on GPU
    alpha:            (k,) tensor
    """
    batch_size, n_unique = unique_ids_batch.shape
    k = log_beta_T.shape[1]

    # Gather log_beta cols for this batch — (batch_size, n_unique, k)
    log_phi = log_beta_T[unique_ids_batch]

    # Warm-start
    log_phi -= log_phi.max(dim=2, keepdim=True).values
    phi = torch.softmax(log_phi, dim=2)                          # (batch_size, n_unique, k)

    # gamma init — (batch_size, k)
    gamma = alpha.unsqueeze(0) + (phi * counts_batch.unsqueeze(2)).sum(dim=1)

    for _ in range(max_iter):
        gamma_old = gamma.clone()

        # E[log theta] — (batch_size, k)
        log_exp_theta = (_digamma_torch(gamma)
                         - _digamma_torch(gamma.sum(dim=1, keepdim=True)))

        # phi update — (batch_size, n_unique, k)
        log_phi = log_beta_T[unique_ids_batch] + log_exp_theta.unsqueeze(1)
        log_phi -= log_phi.max(dim=2, keepdim=True).values
        phi = torch.softmax(log_phi, dim=2)

        # gamma update
        gamma = alpha.unsqueeze(0) + (phi * counts_batch.unsqueeze(2)).sum(dim=1)

        if (gamma - gamma_old).abs().mean() < tol:
            break

    return gamma, phi   # (batch_size, k), (batch_size, n_unique, k)


def _bucket_corpus(corpus_sparse, bucket_size=8):
    """Group docs by unique word count into buckets to minimize padding waste."""
    indexed = sorted(enumerate(corpus_sparse), key=lambda x: len(x[1][0]))
    buckets = []
    for _, group in groupby(indexed, key=lambda x: len(x[1][0]) // bucket_size):
        buckets.append(list(group))
    return buckets



class VariationalLDA2:
    def __init__(self, num_topics, vocab_size, alpha_init=1.0, eta=0.01):
        self.k = num_topics  # Number of topics [4]
        self.v = vocab_size  # Vocabulary size [5]
        # Corpus-level parameters [6]
        self.alpha = np.full(self.k, alpha_init)  # Dirichlet prior [4, 7]
        self.beta = np.random.dirichlet([1.0] * self.v, self.k) # Topic-word matrix [4]
        self.eta = eta # Smoothing parameter for beta [8, 9]

    def _estep_batch(self, bow, log_beta_T, gammas, alpha, beta_numerator, max_iter=100, tol=1e-3):
        """
        bow:        (n_docs, v) CSR — word counts
        log_beta_T: (v, k)
        gammas:     (n_docs, k) — mutated in place, cached across EM iters
        """
        n_docs = bow.shape[0]
        ss_alpha = np.zeros(alpha.shape[0])

        for d in tqdm(range(n_docs), desc="E-step"):
            # sparse row slice — only non-zero vocab entries
            row = bow.getrow(d)
            unique_ids = row.indices
            counts = row.data

            log_phi = log_beta_T[unique_ids]                               # (n_unique, k)
            log_exp_theta = digamma(gammas[d]) - digamma(gammas[d].sum())
            log_phi = log_phi + log_exp_theta
            log_phi -= log_phi.max(axis=1, keepdims=True)
            phi = np.exp(log_phi)
            phi /= phi.sum(axis=1, keepdims=True)
            gammas[d] = alpha + (phi * counts[:, None]).sum(axis=0)

            for _ in range(max_iter - 1):
                gamma_old = gammas[d].copy()
                log_exp_theta = digamma(gammas[d]) - digamma(gammas[d].sum())
                log_phi = log_beta_T[unique_ids] + log_exp_theta
                log_phi -= log_phi.max(axis=1, keepdims=True)
                phi = np.exp(log_phi)
                phi /= phi.sum(axis=1, keepdims=True)
                gammas[d] = alpha + (phi * counts[:, None]).sum(axis=0)
                if np.abs(gammas[d] - gamma_old).mean() < tol:
                    break

            beta_numerator[:, unique_ids] += (phi * counts[:, None]).T
            ss_alpha += digamma(gammas[d]) - digamma(gammas[d].sum())

        return gammas, beta_numerator, ss_alpha

    def train_gpu(self, corpus_word_ids, em_iter=10, tol=1e-4, batch_size=512):
        corpus_sparse = sorted(
            [np.unique(doc, return_counts=True)
            for doc in tqdm(corpus_word_ids, desc="Building sparse corpus")],
            key=lambda x: len(x[0])
        )
        n_docs = len(corpus_sparse)

        # Move beta to GPU once
        log_beta_T = torch.tensor(
            np.ascontiguousarray(np.log(self.beta + 1e-12).T),
            dtype=torch.float32, device=device
        )
        alpha_gpu = torch.tensor(self.alpha, dtype=torch.float32, device=device)

        # Cache gammas on GPU
        gammas_np = np.tile(self.alpha, (n_docs, 1)).astype(np.float32)

        buckets = _bucket_corpus(corpus_sparse, bucket_size=8)

        for i in range(em_iter):
            beta_old = self.beta.copy()
            log_beta_T = torch.tensor(
                np.ascontiguousarray(np.log(self.beta + 1e-12).T),
                dtype=torch.float32, device=device
            )
            beta_numerator = np.full_like(self.beta, self.eta)
            ss_alpha = np.zeros(self.k)

            for bucket in tqdm(buckets, desc=f"EM {i+1}/{em_iter}"):
                # Process bucket in mini-batches
                for start in range(0, len(bucket), batch_size):
                    mini = bucket[start:start + batch_size]
                    doc_indices = [idx for idx, _ in mini]
                    docs = [doc for _, doc in mini]

                    # Pad to same length within mini-batch
                    max_len = max(len(u) for u, _ in docs)
                    pad_ids    = np.zeros((len(docs), max_len), dtype=np.int64)
                    pad_counts = np.zeros((len(docs), max_len), dtype=np.float32)

                    for j, (uid, cnt) in enumerate(docs):
                        pad_ids[j, :len(uid)]    = uid
                        pad_counts[j, :len(cnt)] = cnt

                    # Send to GPU
                    ids_gpu    = torch.tensor(pad_ids,    device=device)
                    counts_gpu = torch.tensor(pad_counts, device=device)

                    with torch.no_grad():
                        gamma_d, phi_d = _estep_batch_gpu(
                            ids_gpu, counts_gpu, log_beta_T, alpha_gpu
                        )

                    # Accumulate beta_numerator — bring phi back to CPU
                    phi_cpu    = phi_d.cpu().numpy()           # (B, max_len, k)
                    counts_cpu = pad_counts                    # (B, max_len)
                    gamma_cpu  = gamma_d.cpu().numpy()

                    for j, (uid, _) in enumerate(docs):
                        n = len(uid)
                        weighted = (phi_cpu[j, :n] * counts_cpu[j, :n, None]).T  # (k, n)
                        beta_numerator[:, uid] += weighted
                        ss_alpha += digamma(gamma_cpu[j]) - digamma(gamma_cpu[j].sum())

                    # Cache updated gammas
                    gammas_np[doc_indices] = gamma_cpu

            self.beta = beta_numerator / beta_numerator.sum(axis=1, keepdims=True)
            self._update_alpha(n_docs, ss_alpha)

            delta = np.abs(self.beta - beta_old).mean()
            print(f"  EM {i+1}: beta delta = {delta:.6f}")

    

    def _update_alpha(self, M, ss_alpha, max_iter=20):
        """
        Newton-Raphson for Dirichlet parameter alpha [12, 16, 20, 21].
        """
        for _ in range(max_iter):
            sum_alpha = np.sum(self.alpha)
            # Gradient [20]
            g = M * (digamma(sum_alpha) - digamma(self.alpha)) + ss_alpha
            # Hessian components [20, 21]
            h = -M * polygamma(1, self.alpha)
            z = M * polygamma(1, sum_alpha)
            
            # Linear-time matrix inversion [12, 21]
            c = np.sum(g / h) / (1.0/z + np.sum(1.0/h))
            step = (g - c) / h

            scale = 1.0
            while np.any(self.alpha - scale * step <= 0):
                scale *= 0.5
            self.alpha -= scale * step

def print_top_topics(model, vocab, n_words=10):
    """
    vocab: list or array of strings where vocab[i] is the word for id i
    """
    topic_dicts = []
    for topic_idx in range(model.k):
        top_ids = np.argsort(model.beta[topic_idx])[::-1][:n_words]
        top_words = [(vocab[i], model.beta[topic_idx][i]) for i in top_ids]
        top_dict = {vocab[i]:model.beta[topic_idx][i] for i in top_ids}
        topic_dicts.append(top_dict)
        words_str = "  ".join(f"{w} ({p:.3f})" for w, p in top_words)
        # print(f"Topic {topic_idx:2d}: {words_str}")
    return topic_dicts
import json 
def export_model(lda_model, word_to_id, topic_dicts, path="model.json"):
    # vocab as list ordered by id
    id_to_word = {v: k for k, v in word_to_id.items()}
    vocab_list  = [id_to_word[i] for i in range(len(id_to_word))]

    payload = {
        "alpha": lda_model.alpha.tolist(),
        "beta":  lda_model.beta.tolist(),   # (k, v)
        "vocab": vocab_list,
        "topic_descs": topic_dicts 
    }
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))   # compact
    print(f"Exported — vocab {len(vocab_list)}, topics {lda_model.k}")
from datasets import load_from_disk
for year in [1995, 2000, 2005, 2010]:
    data = load_from_disk("/home/wsl_default/MIT/6.783/gigaword/gigaword_eng_5/data/nyt_eng/_nyt_1995")
    data = data.filter(lambda x: len(x["text"].split()) > 100)
    data = data.shuffle(seed=42).select(range(100_000))["text"]
    tokenized_data = preprocess_data(data)
    vocab, doc_freq = build_vocab(tokenized_data)
    corpus_word_ids, word_to_id = get_ids(tokenized_data, vocab)

    for NUM_TOPICS in [10,50,100]:
        VOCAB_SIZE = len(word_to_id)

        lda_model = VariationalLDA2(num_topics=NUM_TOPICS, vocab_size=VOCAB_SIZE)
        lda_model.train_gpu(corpus_word_ids, em_iter=10, batch_size=8192)
        topic_dicts = print_top_topics(lda_model, sorted(vocab))
        export_model(lda_model, word_to_id, topic_dicts, path=f"nyt{year}_100k_{NUM_TOPICS}.json")