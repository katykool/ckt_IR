import argparse
from pipeline import load_corpus, build_indices, search, print_results, tokenize
from spellchecker import SpellChecker, build_vocab
from recommender import load_news, build_news_index, recommend


def main():
    parser = argparse.ArgumentParser(
        description="Поиск по чукотскому корпусу с рекомендациями",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "query",
        type=str,
        help="Поисковый запрос на чукотском языке"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["search", "recommend", "all"],
        help=(
            "Режим работы (default: all):\n"
            "  search    — только поиск по корпусу\n"
            "  recommend — только рекомендации из газет\n"
            "  all       — поиск и рекомендации вместе"
        )
    )
    parser.add_argument(
        "--index",
        type=str,
        default="bm25",
        choices=["bm25", "w2v", "fasttext", "ri"],
        help=(
            "Тип индекса (default: bm25):\n"
            "  bm25     — лексический поиск BM25\n"
            "  w2v      — Word2Vec + косинусное сходство\n"
            "  fasttext — FastText + косинусное сходство\n"
            "  ri       — Random Indexing + косинусное сходство"
        )
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Количество результатов (default: 5)"
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        help="Исправить опечатки в запросе перед поиском"
    )
    parser.add_argument(
        "--news_path",
        type=str,
        default="nutenut_news.xlsx",
        help="Путь к файлу газетного корпуса (default: nutenut_news.xlsx)"
    )

    args = parser.parse_args()

   # загрузка корпуса
    doc_texts_ckt, doc_texts_ru = load_corpus()
    doc_tokens = [tokenize(text) for text in doc_texts_ckt]

    # спеллчекер
    query = args.query
    if args.spellcheck:
        spell = SpellChecker(build_vocab(doc_tokens))
        query = spell.correct_query(query)
        print(f"Исправленный запрос: {query}\n")

    # поиск
    if args.mode in ("search", "all"):
        indices = build_indices(doc_texts_ckt)
        print(f"{'='*50}")
        print(f"ПОИСК  |  запрос: {query}  |  индекс: {args.index}")
        print(f"{'='*50}\n")
        results = search(
            query=query,
            index=args.index,
            indices=indices,
            doc_texts_ckt=doc_texts_ckt,
            doc_texts_ru=doc_texts_ru,
            top_k=args.top_k
        )
        print_results(results)

    # рекомендации
    if args.mode in ("recommend", "all"):
        print(f"{'*'*50}")
        print(f"Хочешь узнать подробнее?")
        print(f"{'*'*50}\n")
        news = load_news(args.news_path)
        tfidf, matrix = build_news_index(news)
        recommend(query, news, tfidf, matrix, top_k=3)


if __name__ == "__main__":
    main()