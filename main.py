
from pathlib import Path
import re

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "Input.xlsx"
STOPWORDS_DIR = BASE_DIR / "StopWords"
MASTER_DIR = BASE_DIR / "MasterDictionary"
TEXT_DIR = BASE_DIR / "scraped_test"
OUTPUT_FILE = BASE_DIR / "Output.xlsx"

# First run: process only five articles.
# Change to None after validating the test output.
TEST_LIMIT = None

TEXT_DIR.mkdir(exist_ok=True)

for resource in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(resource, quiet=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUTPUT_COLUMNS = [
    "URL_ID",
    "URL",
    "POSITIVE SCORE",
    "NEGATIVE SCORE",
    "POLARITY SCORE",
    "SUBJECTIVITY SCORE",
    "AVG SENTENCE LENGTH",
    "PERCENTAGE OF COMPLEX WORDS",
    "FOG INDEX",
    "AVG NUMBER OF WORDS PER SENTENCE",
    "COMPLEX WORD COUNT",
    "WORD COUNT",
    "SYLLABLE PER WORD",
    "PERSONAL PRONOUNS",
    "AVG WORD LENGTH",
]


# ==========================================================
# 2. LOAD STOPWORDS AND SENTIMENT DICTIONARIES
# ==========================================================

def load_stopwords(folder):
    words = set()

    for path in folder.glob("*.txt"):
        text = path.read_bytes().decode("cp1252")

        for line in text.splitlines():
            word = line.split("|", 1)[0].strip().lower()

            if word:
                words.add(word)

    return words


def load_sentiment_words(path):
    text = path.read_bytes().decode("cp1252")

    return {
        line.strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    }


stopwords = load_stopwords(STOPWORDS_DIR)

positive_words = (
    load_sentiment_words(MASTER_DIR / "positive-words.txt")
    - stopwords
)

negative_words = (
    load_sentiment_words(MASTER_DIR / "negative-words.txt")
    - stopwords
)

nltk_english_stopwords = set(nltk_stopwords.words("english"))



# ==========================================================
# 3. SCRAPE AND SAVE ARTICLE TEXT
# ==========================================================

def normalize_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",(?=\S)", ", ", text)
    return text.strip()


def extract_article(html):
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article")

    if article is None:
        return "", ""

    title_tag = article.find("h1") or soup.find("h1")

    title = (
        normalize_text(title_tag.get_text(" ", strip=True))
        if title_tag
        else ""
    )

    content = (
        article.select_one(".td-post-content")
        or article.select_one(".entry-content")
        or article
    )

    paragraphs = []

    for paragraph in content.find_all("p"):
        if paragraph.find_parent("p"):
            continue

        text = normalize_text(
            paragraph.get_text(" ", strip=True)
        )

        if not text:
            continue

        # Skip a paragraph consisting only of a URL.
        if re.fullmatch(r"https?://\S+", text):
            continue

        paragraphs.append(text)

    return title, "\n".join(paragraphs)


def scrape_article(session, url):
    response = session.get(url, timeout=30)
    response.raise_for_status()

    title, body = extract_article(response.text)

    if not title or not body:
        raise ValueError("Article title or body is missing.")

    return title, body


# ==========================================================
# 4. CLEAN WORDS
# ==========================================================

def clean_word(token):
    return re.sub(r"^[^\w]+|[^\w]+$", "", token).lower()


def get_sentiment_words(text):
    words = []

    for token in word_tokenize(text.lower()):
        word = clean_word(token)

        if not word:
            continue

        if not re.search(r"[a-z0-9]", word):
            continue

        if word in stopwords:
            continue

        words.append(word)

    return words


def get_analysis_words(text):
    words = []

    for token in word_tokenize(text):
        word = clean_word(token)

        if not word:
            continue

        if not re.search(r"[a-z0-9]", word):
            continue

        if word in nltk_english_stopwords:
            continue

        words.append(word)

    return words


# ==========================================================
# 5. SENTIMENT ANALYSIS
# ==========================================================

def calculate_sentiment(text):
    words = get_sentiment_words(text)

    positive_score = sum(
        word in positive_words for word in words
    )

    negative_score = sum(
        word in negative_words for word in words
    )

    polarity_score = (
        (positive_score - negative_score)
        / (positive_score + negative_score + 0.000001)
    )

    total_cleaned_words = len(get_analysis_words(text))

    subjectivity_score = (
    (positive_score + negative_score)
    / (total_cleaned_words + 0.000001)
)

    return {
        "POSITIVE SCORE": positive_score,
        "NEGATIVE SCORE": negative_score,
        "POLARITY SCORE": polarity_score,
        "SUBJECTIVITY SCORE": subjectivity_score,
    }


# ==========================================================
# 6. SYLLABLE COUNTING
# ==========================================================


def count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())

    if not word:
        return 0

    # Count consecutive vowel groups.
    count = len(re.findall(r"[aeiouy]+", word))

    # Handle common -ed endings.
    if word.endswith("ed") and len(word) > 3:
        # Usually, -ed is not an extra syllable,
        # except in endings pronounced as a separate syllable.
        if word[-3] not in "td":
            count -= 1

    # Handle -es endings.
    elif word.endswith("es") and len(word) > 3:
        if word[-3] not in "szhx":
            count -= 1

    return max(1, count)

# ==========================================================
# 7. PERSONAL PRONOUNS
# ==========================================================

def count_personal_pronouns(text):
    pattern = re.compile(
        r"\b(?:I|we|my|ours|us)\b",
        re.IGNORECASE,
    )

    count = 0

    for match in pattern.finditer(text):
        # Exclude uppercase US when it denotes the country.
        if match.group() == "US":
            continue

        count += 1

    return count


# ==========================================================
# 8. CALCULATE ALL ANALYSIS VARIABLES
# ==========================================================

def analyze_text(text):
    sentiment = calculate_sentiment(text)

    words = get_analysis_words(text)
    sentences = sent_tokenize(text)

    word_count = len(words)
    sentence_count = len(sentences)

    complex_word_count = sum(
        count_syllables(word) > 2 for word in words
    )

    total_syllables = sum(
        count_syllables(word) for word in words
    )

    total_characters = sum(len(word) for word in words)

    avg_sentence_length = (
        word_count / sentence_count if sentence_count else 0.0
    )

    percentage_complex_words = (
        complex_word_count / word_count if word_count else 0.0
    )

    fog_index = 0.4 * (
        avg_sentence_length + percentage_complex_words
    )

    syllable_per_word = (
        total_syllables / word_count if word_count else 0.0
    )

    avg_word_length = (
        total_characters / word_count if word_count else 0.0
    )

    return {
        **sentiment,
        "AVG SENTENCE LENGTH": avg_sentence_length,
        "PERCENTAGE OF COMPLEX WORDS": percentage_complex_words,
        "FOG INDEX": fog_index,
        "AVG NUMBER OF WORDS PER SENTENCE": avg_sentence_length,
        "COMPLEX WORD COUNT": complex_word_count,
        "WORD COUNT": word_count,
        "SYLLABLE PER WORD": syllable_per_word,
        "PERSONAL PRONOUNS": count_personal_pronouns(text),
        "AVG WORD LENGTH": avg_word_length,
    }


# ==========================================================
# 9. MAIN PIPELINE
# ==========================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    required_columns = {"URL_ID", "URL"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Input.xlsx must contain URL_ID and URL columns."
        )

    if TEST_LIMIT is not None:
        df = df.head(TEST_LIMIT)

    results = []
    failures = []

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for _, row in df.iterrows():
            url_id = str(row["URL_ID"]).strip()
            url = str(row["URL"]).strip()

            if url_id.lower() in ("", "nan"):
                print("FAILED: Missing URL_ID")
                failures.append("Missing URL_ID")
                continue

            if url.lower() in ("", "nan"):
                print(f"FAILED: {url_id} — missing URL")
                failures.append(url_id)
                continue

            try:
                title, body = scrape_article(session, url)

                article_text = f"{title}\n\n{body}"

                text_file = TEXT_DIR / f"{url_id}.txt"
                text_file.write_text(
                    article_text,
                    encoding="utf-8",
                )

                metrics = analyze_text(article_text)

                results.append({
                    "URL_ID": row["URL_ID"],
                    "URL": url,
                    **metrics,
                })

                print(f"SUCCESS: {url_id}")

            except (
                requests.RequestException,
                ValueError,
                OSError,
            ) as error:
                print(f"FAILED: {url_id} — {error}")
                failures.append(url_id)

    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    output_df.to_excel(OUTPUT_FILE, index=False)

    print("\n--- PIPELINE SUMMARY ---")
    print(f"Articles selected: {len(df)}")
    print(f"Articles analyzed: {len(results)}")
    print(f"Articles failed: {len(failures)}")
    print(f"Excel output: {OUTPUT_FILE}")

    if failures:
        print("Failed IDs:")
        for url_id in failures:
            print(f"  - {url_id}")


if __name__ == "__main__":
    main()
