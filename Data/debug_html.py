from test_scraper import scrape_book_details
import sys
url = "https://www.goodreads.com/book/show/2767052-the-hunger-games"
details = scrape_book_details(url)
print(details)