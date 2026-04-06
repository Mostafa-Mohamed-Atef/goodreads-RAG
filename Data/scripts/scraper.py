import requests
from bs4 import BeautifulSoup
import time
import csv
import json

BASE_URL = "https://www.goodreads.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ---------------------------
# Extract metadata (Format, ISBN, etc.)
# ---------------------------
def extract_book_metadata(soup):
    metadata = {}
    try:
        # Get format and language from JSON-LD
        script_tag = soup.find("script", type="application/ld+json")
        if script_tag:
            import json
            data = json.loads(script_tag.string)
            if isinstance(data, list):
                data = data[0]

            pages = data.get("numberOfPages", "")
            book_format = data.get("bookFormat", "")
            if pages or book_format:
                metadata["format"] = f"{pages} pages, {book_format}".strip(", ")
            
            metadata["language"] = data.get("inLanguage", "")
            
        # Get published date
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
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        # ⭐ Rating
        rating = soup.find("div", class_="RatingStatistics__rating")
        rating = rating.text.strip() if rating else None

        # 📊 Counts
        ratings_count = soup.find("span", {"data-testid": "ratingsCount"})
        ratings_count = ratings_count.text.strip() if ratings_count else None

        reviews_count = soup.find("span", {"data-testid": "reviewsCount"})
        reviews_count = reviews_count.text.strip() if reviews_count else None

        # 📝 Description
        desc_div = soup.find("div", {"data-testid": "description"})
        description = desc_div.get_text(separator=" ", strip=True) if desc_div else None

        # 📦 Metadata
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
        return None


# ---------------------------
# Scrape top 100 books list
# ---------------------------
def scrape_books():
    books = []

    for page in range(1, 3):  # 5 pages = 100 books
        print(f"\n📄 Scraping list page {page}...")

        url = f"https://www.goodreads.com/list/show/1.Best_Books_Ever?page={page}"
        response = requests.get(url, headers=HEADERS)

        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.find_all("tr", itemtype="http://schema.org/Book")

        for row in rows:
            try:
                rank = row.find("td", class_="number").text.strip()

                title_tag = row.find("a", class_="bookTitle")
                title = title_tag.text.strip()
                book_url = BASE_URL + title_tag["href"]

                author = row.find("a", class_="authorName").text.strip()

                book_data = {
                    "rank": rank,
                    "title": title,
                    "author": author,
                    "url": book_url
                }

                print(f"📖 {rank} - {title}")

                # 🔥 Scrape book details
                details = scrape_book_details(book_url)

                if details:
                    book_data.update(details)

                books.append(book_data)

                time.sleep(2)  # VERY IMPORTANT (avoid blocking)

            except Exception as e:
                print("⚠️ Skipping book:", e)
                continue

    return books


# ---------------------------
# Save functions
# ---------------------------
def save_to_csv(books):
    keys = books[0].keys()

    with open("goodreads_top100_full.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(books)


def save_to_json(books):
    with open("goodreads_top100_full.json", "w", encoding="utf-8") as f:
        json.dump(books, f, indent=4, ensure_ascii=False)


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    books = scrape_books()

    print(f"\n🎉 Total books scraped: {len(books)}")

    save_to_csv(books)
    save_to_json(books)

    print("💾 Data saved successfully!")