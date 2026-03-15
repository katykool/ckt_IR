from collections import Counter
import re

# Алфавит чукотского языка (кириллица + специфические буквы)
CHUKCHI_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяӄӈԓ'

# словарь частот
def build_vocab(doc_tokens):
    all_words = [token for tokens in doc_tokens for token in tokens]
    return Counter(all_words)

# операции
def splits(word):
    """Все возможные разбиения слова на пару (левая, правая часть)."""
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


def delete_letter(word, split):
    """Удаление одной буквы."""
    return [L + R[1:] for L, R in split if R]


def transpose_letters(word, split):
    """Перестановка двух соседних букв."""
    return [L + R[1] + R[0] + R[2:] for L, R in split if len(R) > 1]


def replace_letter(word, split, alphabet=CHUKCHI_ALPHABET):
    """Замена одной буквы на любую из алфавита."""
    return [L + c + R[1:] for L, R in split if R for c in alphabet if c != R[0]]


def insert_letter(word, split, alphabet=CHUKCHI_ALPHABET):
    """Вставка одной буквы в любое место."""
    return [L + c + R for L, R in split for c in alphabet]


def edits1(word, alphabet=CHUKCHI_ALPHABET):
    """Все слова на расстоянии Левенштейна 1 от word."""
    sp = splits(word)
    return set(
        delete_letter(word, sp) +
        transpose_letters(word, sp) +
        replace_letter(word, sp, alphabet) +
        insert_letter(word, sp, alphabet)
    )


def edits2(word):
    """Все слова на расстоянии Левенштейна 2 от word."""
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))



class SpellChecker:

    def __init__(self, vocab):
        self.vocab = vocab
        self.total = sum(vocab.values())

    def prob(self, word):
        return self.vocab.get(word, 0) / self.total

    def known(self, words): # вернуть подмножество слов, которые есть в словаре
        return set(w for w in words if w in self.vocab)

    # Кандидаты на исправление:
    # сначала точное совпадение, потом edit1, потом edit2.
    # Если ничего не нашли — оставляем слово как есть.
    def candidates(self, word):
        return (
            self.known({word}) or
            self.known(edits1(word)) or
            self.known(edits2(word)) or
            {word}
        )

    # Наиболее вероятное исправление одного слова
    def correct_word(self, word):
        return max(self.candidates(word), key=self.prob)

    # Исправляет все слова в запросе
    def correct_query(self, query):

        tokens = re.findall(r'[\w\u0400-\u04FF\u0500-\u052F]+', query.lower())
        corrected = [self.correct_word(t) for t in tokens]

        changes = [(o, c) for o, c in zip(tokens, corrected) if o != c]
        if changes:
            print("Исправления:")
            for orig, corr in changes:
                print(f"  {orig} → {corr}")
        else:
            print("Опечаток не найдено.")

        return " ".join(corrected)
