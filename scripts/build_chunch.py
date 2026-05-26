import json
import os
from pathlib import Path

RAW_DIRS = [ "data/raw/en" , "data/raw/hi"]

OUTPUT_FILE = "data/processed/chunks.jsonl"

BAD_PASSAGES = {
    "gallery",
    "references",
    "external links",
    "see also"
}

Path("data/processed").mkdir(parents=True, exist_ok=True)

total_chunks = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for raw_dir in RAW_DIRS:

        for filepath in Path(raw_dir).glob("*.json"):

            with open(filepath, "r", encoding="utf-8") as f:
                article = json.load(f)

            metadata = {
                "article_id": article["article_id"],
                "title": article["title"],
                "language": article["language"],
                "url": article["url"],
                "source_file": str(filepath)
            }

            for passage in article["passages"]:

                passage = passage.strip()

                if (
                    len(passage) < 20
                    or passage.lower() in BAD_PASSAGES
                ):
                    continue

                doc = {
                    "text": passage,
                    "metadata": metadata
                }

                outfile.write(
                    json.dumps(doc, ensure_ascii=False)
                    + "\n"
                )

                total_chunks += 1

print(f"Saved {total_chunks} chunks.")
