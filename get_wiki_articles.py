import wikipediaapi
import requests
import os
import json
import uuid
import time

# Languages
LANGUAGES = ["en", "hi"]

# Target articles per language
TARGET_ARTICLES = 100

BASE_DIR = "data/raw"


import requests

import requests

def get_random_title(lang):
    """Fetch random Wikipedia title safely"""

    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code == 403:
            print(f"[{lang}] Blocked (403). Retrying...")
            return None

        if response.status_code != 200:
            print(f"[{lang}] HTTP Error:", response.status_code)
            return None

        data = response.json()

        return data.get("title")

    except requests.exceptions.RequestException as e:
        print(f"[{lang}] Request failed:", e)
        return None


def extract_passages(text):
    """Split text into paragraphs (no length filtering)"""

    paragraphs = text.split("\n")

    clean = []

    for p in paragraphs:
        p = p.strip()

        if len(p) > 0:
            clean.append(p)

    return clean


def save_article(article_data, lang):
    """Save article JSON"""

    folder = os.path.join(BASE_DIR, lang)

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(
        folder,
        f"{article_data['article_id']}.json"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(article_data, f, ensure_ascii=False, indent=2)


def collect_language(lang):

    print(f"\nStarting collection for: {lang}")

    wiki = wikipediaapi.Wikipedia(
        user_agent = "mragdata (redashcatch@gmail.com)",
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    saved_count = 0
    attempts = 0

    while saved_count < TARGET_ARTICLES:

        attempts += 1

        try:

            title = get_random_title(lang)

            if not title:
                continue

            page = wiki.page(title)

            # Handle missing pages
            if not page.exists():
                print(f"[{lang}] Page not found:", title)
                continue

            passages = extract_passages(page.text)

            if len(passages) == 0:
                print(f"[{lang}] No passages:", title)
                continue

            article_id = f"{lang}_{uuid.uuid4().hex[:8]}"

            article_data = {
                "article_id": article_id,
                "language": lang,
                "title": page.title,
                "wikipedia_page_id": page.pageid,
                "url": page.fullurl,
                "passages": passages
            }

            save_article(article_data, lang)

            saved_count += 1

            print(
                f"[{lang}] Saved {saved_count}/{TARGET_ARTICLES} | {title}"
            )

            # Small delay to avoid API overload
            time.sleep(0.5)

        except Exception as e:

            print(f"[{lang}] Error processing article:", e)
            continue

    print(f"Completed {lang} collection.")


def main():

    for lang in LANGUAGES:

        collect_language(lang)

    print("\nAll collection complete.")


if __name__ == "__main__":
    main()
