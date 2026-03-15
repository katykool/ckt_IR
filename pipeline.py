from datasets import load_dataset
from gensim.models import Word2Vec, FastText
from rank_bm25 import BM25Okapi
from collections import defaultdict
import numpy as np
import time
import re

# загрузка данных
def load_corpus():
    """Загружает и фильтрует параллельный корпус."""
    parallel = load_dataset("HSE-Chukchi-NLP/russian-chukchi-parallel-corpora")
    df = parallel["train"]

    valid = [
        (ckt, ru)
        for ckt, ru, score in zip(df["ckt"], df["ru"], df["score"])
        if ckt and ru
        and len(ckt.split()) > 1
        and score > 0.6
    ]
    doc_texts_ckt, doc_texts_ru = zip(*valid)
    return list(doc_texts_ckt), list(doc_texts_ru)

# токенизация -- просто по пробелам
def tokenize(text):
    text = text.lower()
    return re.findall(r'[\w\u0400-\u04FF\u0500-\u052F]+', text) # расширенная кириллица

# word2vec
def text_to_vector_w2v(tokens, model):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(vec) # нормализация
    return vec / norm if norm > 0 else vec

# Random Indexing 
def text_to_vector_ri(tokens, word_vectors, dim):
    vecs = [word_vectors[t] for t in tokens if t in word_vectors]
    if not vecs:
        return np.zeros(dim)
    vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_sim(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return np.dot(v1, v2) / denom


# ИНДЕКСИРОВАНИЕ
# BM25
def build_bm25(doc_tokens):
    return BM25Okapi(doc_tokens)

# word2vec обучение
def build_w2v(doc_tokens):
    model = Word2Vec(
        sentences=doc_tokens,
        vector_size=50,
        window=3,
        min_count=1,
        workers=4,
        epochs=10
    )
    doc_vectors = [text_to_vector_w2v(tokens, model) for tokens in doc_tokens]
    return model, doc_vectors

# FastText обучение
def build_fasttext(doc_tokens):
    model = FastText(
        sentences=doc_tokens,
        vector_size=50,
        window=3,
        min_count=1,
        workers=4,
        epochs=10
    )
    doc_vectors = [text_to_vector_w2v(tokens, model) for tokens in doc_tokens]
    return model, doc_vectors

# Random Indexing
def build_ri(doc_tokens, dim=1000, nonzero=6):
    def make_signature(): # назначается случайный разреженный вектор
        sig = np.zeros(dim)
        indices = np.random.choice(dim, nonzero, replace=False)
        sig[indices[:nonzero // 2]] = 1
        sig[indices[nonzero // 2:]] = -1
        return sig

    doc_signatures = [make_signature() for _ in range(len(doc_tokens))]

    word_vectors = defaultdict(lambda: np.zeros(dim)) # строятся векторы слов. слово накапливает в себе doc_signatures всех документов где встречалось
    for i, tokens in enumerate(doc_tokens):
        for token in set(tokens):
            word_vectors[token] += doc_signatures[i]

    doc_vectors = [text_to_vector_ri(tokens, word_vectors, dim) for tokens in doc_tokens] # строятся векторы документов
    return word_vectors, doc_vectors, dim

# построение всех индексов
def build_indices(doc_texts_ckt):
    doc_tokens = [tokenize(text) for text in doc_texts_ckt]

    print("BM25...")
    bm25 = build_bm25(doc_tokens)

    print("Word2Vec...")
    w2v_model, doc_vectors_w2v = build_w2v(doc_tokens)

    print("FastText...")
    ft_model, doc_vectors_ft = build_fasttext(doc_tokens)

    print("Random Indexing...")
    ri_word_vectors, doc_vectors_ri, ri_dim = build_ri(doc_tokens)

    return {
        "doc_tokens": doc_tokens,
        "bm25": bm25,
        "w2v_model": w2v_model,
        "doc_vectors_w2v": doc_vectors_w2v,
        "ft_model": ft_model,
        "doc_vectors_ft": doc_vectors_ft,
        "ri_word_vectors": ri_word_vectors,
        "doc_vectors_ri": doc_vectors_ri,
        "ri_dim": ri_dim,
    }

# ПОИСК
# bm25
def search_bm25(query, indices, doc_texts_ckt, doc_texts_ru, top_k=5):
    query_tokens = tokenize(query)
    start = time.time()
    scores = indices["bm25"].get_scores(query_tokens)
    elapsed = time.time() - start
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"Время поиска BM25: {elapsed:.4f} сек\n")
    return [(doc_texts_ckt[i], doc_texts_ru[i], scores[i]) for i in top_idx]

# word 2 vec
def search_w2v(query, indices, doc_texts_ckt, doc_texts_ru, top_k=5):
    query_vec = text_to_vector_w2v(tokenize(query), indices["w2v_model"])
    start = time.time()
    scores = [cosine_sim(query_vec, dv) for dv in indices["doc_vectors_w2v"]]
    elapsed = time.time() - start
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"Время поиска W2V: {elapsed:.4f} сек\n")
    return [(doc_texts_ckt[i], doc_texts_ru[i], scores[i]) for i in top_idx]

# ri
def search_ri(query, indices, doc_texts_ckt, doc_texts_ru, top_k=5):
    """Поиск с Random Indexing."""
    query_vec = text_to_vector_ri(
        tokenize(query), indices["ri_word_vectors"], indices["ri_dim"]
    )
    start = time.time()
    scores = [cosine_sim(query_vec, dv) for dv in indices["doc_vectors_ri"]]
    elapsed = time.time() - start
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"Время поиска RI: {elapsed:.4f} сек\n")
    return [(doc_texts_ckt[i], doc_texts_ru[i], scores[i]) for i in top_idx]

# fasttext
def search_ft(query, indices, doc_texts_ckt, doc_texts_ru, top_k=5):
    """Поиск с FastText."""
    query_vec = text_to_vector_w2v(tokenize(query), indices["ft_model"])
    start = time.time()
    scores = [cosine_sim(query_vec, dv) for dv in indices["doc_vectors_ft"]]
    elapsed = time.time() - start
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"Время поиска FastText: {elapsed:.4f} сек\n")
    return [(doc_texts_ckt[i], doc_texts_ru[i], scores[i]) for i in top_idx]

def search(query, index, indices, doc_texts_ckt, doc_texts_ru, top_k=5):
    """
    query        : строка запроса на чукотском
    index        : "bm25" | "w2v" | "ri" | "fasttext"
    indices      : словарь с индексами из build_indices()
    doc_texts_ckt: список чукотских документов
    doc_texts_ru : список русских переводов
    top_k        : количество результатов
    """
    if index == "bm25":
        return search_bm25(query, indices, doc_texts_ckt, doc_texts_ru, top_k)
    elif index == "w2v":
        return search_w2v(query, indices, doc_texts_ckt, doc_texts_ru, top_k)
    elif index == "ri":
        return search_ri(query, indices, doc_texts_ckt, doc_texts_ru, top_k)
    elif index == "fasttext":
        return search_ft(query, indices, doc_texts_ckt, doc_texts_ru, top_k)
    else:
        raise ValueError(f"Неизвестный индекс: {index}. Выберите bm25, w2v, ri или fasttext.")


# вывод результатов
def print_results(results):
    for ckt, ru, score in results:
        print(f"[{score:.4f}]")
        print(f"  ckt: {ckt}")
        print(f"  ru:  {ru}")
        print()
