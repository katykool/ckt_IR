import time
import openpyxl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline import tokenize

# газетный корпус
def load_news(path="nutenut_news.xlsx"):
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    news = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        idx, link, date, author, title, post_text, ckt_text, rus_text = row
        if ckt_text:
            news.append({
                "title": title,
                "ckt_text": ckt_text,
                "rus_text": rus_text,
            })
    return news

# построение индексов TF-IDF
def build_news_index(news):
    news_ckt_texts = [n["ckt_text"] for n in news]
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    matrix = tfidf.fit_transform(news_ckt_texts)
    # print(f"TF-IDF матрица новостей: {matrix.shape}")
    return tfidf, matrix

def recommend(query, news, tfidf, matrix, top_k=3):
    """
    query  : строка запроса на чукотском
    news   : список статей из load_news()
    tfidf  : обученный TfidfVectorizer из build_news_index()
    matrix : TF-IDF матрица статей из build_news_index()
    top_k  : количество рекомендаций
    """
    query_vec = tfidf.transform([query])
    start = time.time()
    scores = cosine_similarity(query_vec, matrix)[0]
    elapsed = time.time() - start

    top_idx = scores.argsort()[::-1][:top_k]

    print(f"Запрос: {query}")
    print(f"Время: {elapsed:.4f} сек\n")
    print("Похожие статьи из газеты «Крайний Север»:")
    for i in top_idx:
        rus_preview = (news[i]['rus_text'] or '')[:120]
        print(f"  [{scores[i]:.4f}] {news[i]['title']}")
        print(f"           {rus_preview}...")
    print()
