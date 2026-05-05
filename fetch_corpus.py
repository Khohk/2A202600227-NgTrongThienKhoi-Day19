"""
Step 1: Fetch Tech Company Corpus
- Primary source: GoodWiki (HuggingFace)
- Fallback: Wikipedia API for missing companies
"""

import os
import time
import wikipedia
from datasets import load_dataset

TARGETS = [
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "Mistral AI",
    "Cohere",
    "Nvidia",
    "Microsoft",
    "Meta Platforms",
    "Hugging Face",
    "Amazon Web Services",
]

OUTPUT_DIR = "wiki_md"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Phase 1: GoodWiki ────────────────────────────────────────────────────────

def fetch_from_goodwiki(targets: list[str]) -> set[str]:
    print("Loading GoodWiki dataset (may take a few minutes first time)...")
    ds = load_dataset("euirim/goodwiki", split="train")

    target_set = set(targets)
    found = set()

    for item in ds:
        if item["title"] in target_set:
            filename = item["title"].replace(" ", "_") + ".md"
            path = os.path.join(OUTPUT_DIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {item['title']}\n\n")
                if item.get("description"):
                    f.write(f"> {item['description']}\n\n")
                f.write(item["markdown"])
            print(f"  [GoodWiki] Saved: {filename}")
            found.add(item["title"])

        if found == target_set:
            break  # early exit khi đã đủ

    return found


# ── Phase 2: Wikipedia API fallback ─────────────────────────────────────────

WIKI_FALLBACK_MAP = {
    "Google DeepMind": "Google DeepMind",
    "Mistral AI": "Mistral AI",
    "Cohere": "Cohere (company)",
    "Hugging Face": "Hugging Face",
    "Anthropic": "Anthropic",
    "Amazon Web Services": "Amazon Web Services",
}

def fetch_from_wikipedia(missing: set[str]) -> set[str]:
    wikipedia.set_lang("en")
    fetched = set()

    for company in missing:
        search_title = WIKI_FALLBACK_MAP.get(company, company)
        try:
            page = wikipedia.page(search_title, auto_suggest=False)
            filename = company.replace(" ", "_") + ".md"
            path = os.path.join(OUTPUT_DIR, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# {company}\n\n")
                f.write(page.content)
            print(f"  [Wikipedia] Saved: {filename}")
            fetched.add(company)
            time.sleep(0.5)  # polite rate limit
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  [Wikipedia] Disambiguation for '{company}': {e.options[:3]}")
        except wikipedia.exceptions.PageError:
            print(f"  [Wikipedia] Page not found: '{search_title}'")
        except Exception as e:
            print(f"  [Wikipedia] Error for '{company}': {e}")

    return fetched


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Target companies: {len(TARGETS)}\n")

    # Phase 1
    found = fetch_from_goodwiki(TARGETS)
    print(f"\nGoodWiki: {len(found)}/{len(TARGETS)} found")

    # Phase 2
    missing = set(TARGETS) - found
    if missing:
        print(f"\nFalling back to Wikipedia API for: {missing}")
        fetched = fetch_from_wikipedia(missing)
        still_missing = missing - fetched
        if still_missing:
            print(f"\n[WARNING] Could not fetch: {still_missing}")
    else:
        print("All companies found in GoodWiki!")

    # Summary
    saved = os.listdir(OUTPUT_DIR)
    print(f"\nDone. {len(saved)} files saved in '{OUTPUT_DIR}/':")
    for f in sorted(saved):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"  {f} ({size_kb} KB)")


if __name__ == "__main__":
    main()
