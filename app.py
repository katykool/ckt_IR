from flask import Flask, render_template, request, redirect, url_for
from pipeline import load_corpus, build_indices, search, tokenize
from spellchecker import SpellChecker, build_vocab
from recommender import load_news, build_news_index, recommend as recommend_news
import time

app = Flask(__name__)

print("Загружаем корпус...")
doc_texts_ckt, doc_texts_ru = load_corpus()
doc_tokens = [tokenize(text) for text in doc_texts_ckt]
indices = build_indices(doc_texts_ckt)
spell = SpellChecker(build_vocab(doc_tokens))

print("Загружаем газетный корпус...")
news = load_news("nutenut_news.xlsx")
tfidf, matrix = build_news_index(news)

print("Готово!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET", "POST"])
def search_page():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        index = request.form.get("index", "bm25")
        top_k = int(request.form.get("top_k", 5))
        spellcheck = request.form.get("spellcheck") == "on"
        mode = request.form.get("mode", "all")

        if not query:
            return render_template("search.html", error="Введите запрос")

        return redirect(url_for(
            "results_page",
            query=query,
            index=index,
            top_k=top_k,
            spellcheck=spellcheck,
            mode=mode
        ))

    return render_template("search.html")


@app.route("/results")
def results_page():
    query = request.args.get("query", "").strip()
    index = request.args.get("index", "bm25")
    top_k = int(request.args.get("top_k", 5))
    spellcheck = request.args.get("spellcheck") == "True"
    mode = request.args.get("mode", "all")

    if not query:
        return redirect(url_for("search_page"))

    corrected_query = query
    corrections = []

    # спеллчекинг
    if spellcheck:
        tokens = tokenize(query)
        corrected_tokens = [spell.correct_word(t) for t in tokens]
        corrections = [(o, c) for o, c in zip(tokens, corrected_tokens) if o != c]
        corrected_query = " ".join(corrected_tokens)

    # поиск
    search_results = []
    search_time = None
    if mode in ("search", "all"):
        start = time.time()
        raw = search(
            query=corrected_query,
            index=index,
            indices=indices,
            doc_texts_ckt=doc_texts_ckt,
            doc_texts_ru=doc_texts_ru,
            top_k=top_k
        )
        search_time = round(time.time() - start, 4)
        search_results = [
            {"ckt": ckt, "ru": ru, "score": round(float(score), 4)}
            for ckt, ru, score in raw
        ]

    # рекомендации
    rec_results = []
    rec_time = None
    if mode in ("recommend", "all"):
        start = time.time()
        scores_arr = tfidf.transform([corrected_query])
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(scores_arr, matrix)[0]
        rec_time = round(time.time() - start, 4)
        top_idx = scores.argsort()[::-1][:3]
        rec_results = [
            {
                "title": news[i]["title"],
                "rus_text": (news[i]["rus_text"] or "")[:200],
                "ckt_full": news[i]["ckt_text"] or "",
                "rus_full": news[i]["rus_text"] or "",
                "score": round(float(scores[i]), 4)
            }
            for i in top_idx
        ]

    return render_template(
        "results.html",
        query=query,
        corrected_query=corrected_query,
        corrections=corrections,
        index=index,
        mode=mode,
        search_results=search_results,
        search_time=search_time,
        rec_results=rec_results,
        rec_time=rec_time,
    )


if __name__ == "__main__":
    app.run(debug=True)
