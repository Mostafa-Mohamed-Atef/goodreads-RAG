import requests
from bs4 import BeautifulSoup
import time
import csv
import json

BASE_URL = "https://www.goodreads.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ---------------------------
# Extract metadata (Format, ISBN, etc.)
# ---------------------------
def extract_book_metadata(soup):
    metadata = {}
    try:
        script_tag = soup.find("script", type="application/ld+json")
        if script_tag:
            data = json.loads(script_tag.string)
            if isinstance(data, list):
                data = data[0]

            pages = data.get("numberOfPages", "")
            book_format = data.get("bookFormat", "")
            if pages or book_format:
                metadata["format"] = f"{pages} pages, {book_format}".strip(", ")

            metadata["language"] = data.get("inLanguage", "")

        pub_info = soup.find("p", {"data-testid": "publicationInfo"})
        if pub_info:
            metadata["published"] = pub_info.text.replace("First published", "").strip()

    except Exception as e:
        print("⚠️ Metadata error:", e)

    return metadata


# ---------------------------
# Scrape individual book page
# ---------------------------
def scrape_book_details(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        rating = soup.find("div", class_="RatingStatistics__rating")
        rating = rating.text.strip() if rating else None

        ratings_count = soup.find("span", {"data-testid": "ratingsCount"})
        ratings_count = ratings_count.text.strip() if ratings_count else None

        reviews_count = soup.find("span", {"data-testid": "reviewsCount"})
        reviews_count = reviews_count.text.strip() if reviews_count else None

        desc_div = soup.find("div", {"data-testid": "description"})
        description = desc_div.get_text(separator=" ", strip=True) if desc_div else None

        metadata = extract_book_metadata(soup)

        return {
            "rating": rating,
            "ratings_count": ratings_count,
            "reviews_count": reviews_count,
            "description": description,
            **metadata
        }

    except Exception as e:
        print("❌ Error scraping book page:", e)
        return {}


# ---------------------------
# Scrape Arabic books search results
# The search URL structure:
# https://www.goodreads.com/search?query=arabic&tab=books&page=N
# ---------------------------
def scrape_arabic_books(total_pages=20):
    """
    Scrapes Arabic books from Goodreads search results.
    Each page returns ~20 books, so 5 pages ≈ 100 books.
    Increase total_pages to scrape more.
    """
    books = []
    rank = 1

    for page in range(1, total_pages + 1):
        print(f"\n📄 Scraping search page {page}/{total_pages}...")

        url = (
            f"https://www.goodreads.com/list/show/3865.Recommended_Arabic_Books"
        )

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to fetch page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Search results use <tr> with class "bookTitle" links inside <td class="title">
        rows = soup.select("table.tableList tr[itemtype='http://schema.org/Book']")

        if not rows:
            print("⚠️  No book rows found — page structure may have changed or results ended.")
            break

        for row in rows:
            try:
                title_tag = row.find("a", class_="bookTitle")
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                book_url = BASE_URL + title_tag["href"]

                author_tag = row.find("a", class_="authorName")
                author = author_tag.get_text(strip=True) if author_tag else "Unknown"

                # Rating shown in search results
                avg_rating_tag = row.find("span", class_="minirating")
                avg_rating_text = avg_rating_tag.get_text(strip=True) if avg_rating_tag else ""

                book_data = {
                    "rank": rank,
                    "title": title,
                    "author": author,
                    "search_rating_info": avg_rating_text,
                    "url": book_url,
                }

                print(f"  📖 #{rank} - {title} | {author}")

                # Scrape full details from the book's own page
                details = scrape_book_details(book_url)
                book_data.update(details)

                books.append(book_data)
                rank += 1

                time.sleep(2)  # Be polite to Goodreads

            except Exception as e:
                print(f"⚠️  Skipping a book due to error: {e}")
                continue

    return books


# ---------------------------
# Save functions
# ---------------------------
def save_to_csv(books, filename="goodreads_arabic_books.csv"):
    if not books:
        print("No books to save.")
        return
    keys = books[0].keys()
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig for Excel Arabic support
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(books)
    print(f"✅ CSV saved: {filename}")


def save_to_json(books, filename="goodreads_arabic_books.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(books, f, indent=4, ensure_ascii=False)
    print(f"✅ JSON saved: {filename}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Change total_pages to scrape more books (each page ≈ 20 books)
    books = scrape_arabic_books(total_pages=5)

    print(f"\n🎉 Total books scraped: {len(books)}")

    if books:
        save_to_csv(books)
        save_to_json(books)
        print("💾 Data saved to goodreads_arabic_books.csv and .json")
    else:
        print("⚠️  No books were scraped. Check the URL or your network connection.")