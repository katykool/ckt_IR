import argparse
from pipeline import load_corpus, build_indices, search, print_results


def main():
    parser = argparse.ArgumentParser(
        description="Поиск по чукотскому корпусу",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "query",
        type=str,
        help="Поисковый запрос на чукотском языке"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="bm25",
        choices=["bm25", "w2v", "ri", "fasttext"],
        help=(
            "Тип индекса (default: bm25):\n"
            "  bm25 -- лексический поиск BM25\n"
            "  w2v  -- Word2Vec + косинусное сходство\n"
            "  ri -- Random Indexing + косинусное сходство\n"
            "  fasttext -- FastText + косинусное сходство"
        )
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Количество результатов (default: 5)"
    )

    args = parser.parse_args()

    doc_texts_ckt, doc_texts_ru = load_corpus()
    print(f"Предложений: {len(doc_texts_ckt)}")

    indices = build_indices(doc_texts_ckt)

    print(f"\nЗапрос: {args.query}")
    print(f"Индекс: {args.index}\n")

    results = search(
        query=args.query,
        index=args.index,
        indices=indices,
        doc_texts_ckt=doc_texts_ckt,
        doc_texts_ru=doc_texts_ru,
        top_k=args.top_k
    )
    print_results(results)


if __name__ == "__main__":
    main()
