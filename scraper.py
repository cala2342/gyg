from seleniumbase import SB
import time
import csv
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import random


class Config:
    MAX_SHOW_MORE_CLICKS = 50  # Limit for "show more" clicks (GetYourGuide)
    MAX_PAGES = 50  # Maximum pages to scrape (Viator and Klook)
    OUTPUT_CSV_FILE = "activity_details.csv"  # CSV output file name
    URL_CONFIGS = [
        {
            "region": "europe",
            "url": "https://www.getyourguide.com/europa-l207848?date_from=2025-04-01&date_to=2025-04-30&price_range=50-100&reviewRatingRanges=3.5&dur=0&locale_autoredirect_optout=true",
            "source": "getyourguide"
        },
        {
            "region": "usa",
            "url": "https://www.getyourguide.com/usa-l168990?date_from=2025-04-01&date_to=2025-04-30&price_range=50-100&reviewRatingRanges=3.5&dur=0&locale_autoredirect_optout=true",
            "source": "getyourguide"
        },
        {
            "region": "europe",
            "url": "https://www.viator.com/Europe/d6-ttd",
            "source": "viator"
        },
        {
            "region": "usa",
            "url": "https://www.viator.com/USA/d77-ttd",
            "source": "viator"
        },
        {
            "region": "usa",
            "url": "https://www.klook.com/en-US/coureg/28-usa-things-to-do/",
            "source": "klook"
        }
    ]


# --- Scraper Interface ---

class ActivityScraper(ABC):
    @abstractmethod
    def scrape(self) -> List[Dict[str, str]]:
        """Scrapes and parses activity data from the source."""
        pass


# --- GetYourGuide Scraper ---

class GetYourGuideScraper(ActivityScraper):
    def __init__(self, url: str, retries: int = 2):
        self.url = url
        self.retries = retries

    def scrape(self) -> List[Dict[str, str]]:
        html_content = self._fetch_html()
        activities = self._parse_html(html_content)
        if not activities:
            print(f"No activities found for {self.url}.")
        else:
            print(f"Scraped {len(activities)} activities from {self.url}")
        return activities

    def _fetch_html(self) -> Optional[str]:
        for attempt in range(self.retries):
            try:
                with SB(uc=True, headless=True, browser="chrome") as sb:
                    sb.open(self.url)
                    sb.wait_for_element_present("body", timeout=5)

                    click_count = 0
                    previous_count = 0  # Track the previous number of activities

                    while click_count < Config.MAX_SHOW_MORE_CLICKS and sb.is_element_present(".show-more button"):
                        try:
                            sb.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                            time.sleep(1 + random.uniform(0, 1))  # Randomize sleep between 1 and 2 seconds
                            sb.wait_for_element_visible(".show-more button", timeout=7)
                            current_count = len(sb.find_elements("article.vertical-activity-card"))

                            # Print update after each click, showing new activities loaded
                            if click_count > 0:  # Skip the initial load
                                new_activities = current_count - previous_count
                                print(f"After click {click_count}: Loaded {new_activities} new activities (Total: {current_count}) from {self.url}")

                            sb.scroll_to(".show-more button")
                            sb.click(".show-more button")
                            click_count += 1
                            previous_count = current_count  # Update previous count
                            time.sleep(1 + random.uniform(0, 1))  # Randomize sleep between 1 and 2 seconds
                            WebDriverWait(sb.driver, 10).until(
                                lambda driver: len(driver.find_elements("article.vertical-activity-card")) > current_count
                            )
                        except (NoSuchElementException, TimeoutException):
                            # Final count before breaking
                            final_count = len(sb.find_elements("article.vertical-activity-card"))
                            if final_count > previous_count:
                                print(f"After click {click_count}: Loaded {final_count - previous_count} new activities (Total: {final_count}) from {self.url}")
                            break

                    # Print final count if no "show more" button was clicked
                    if click_count == 0:
                        final_count = len(sb.find_elements("article.vertical-activity-card"))
                        print(f"Initial load: Scraped {final_count} activities from {self.url}")

                    return sb.get_page_source()

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {self.url}: {str(e)}")
                if attempt == self.retries - 1:
                    return None
                time.sleep(2 + random.uniform(0, 1))  # Randomize sleep between 2 and 3 seconds
        return None

    def _parse_html(self, html_content: Optional[str]) -> List[Dict[str, str]]:
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        activity_cards = soup.find_all("article", class_="vertical-activity-card")
        activities = []

        for card in activity_cards:
            title_tag = card.find("h3", class_="vertical-activity-card__title")
            title = title_tag.find("span").get_text(strip=True).replace('"', '') if title_tag else "Unknown Activity"

            rating_tag = card.find("div", class_="c-activity-rating__label")
            ratings_count_raw = rating_tag.get_text(strip=True).strip("()").replace(",", "") if rating_tag else "0"
            ratings_count = int(ratings_count_raw) if ratings_count_raw.isdigit() else 0

            rating_score_tag = card.find("div", class_="c-user-rating")
            rating_score = rating_score_tag["aria-label"].split(" out of ")[0] if rating_score_tag and "aria-label" in rating_score_tag.attrs else "N/A"

            activities.append({
                "title": title,
                "ratings_count": str(ratings_count),
                "rating_score": rating_score
            })

        return activities


# --- Viator Scraper ---

class ViatorScraper(ActivityScraper):
    def __init__(self, base_url: str, max_pages: int = Config.MAX_PAGES):
        self.base_url = base_url
        self.max_pages = max_pages

    def scrape(self) -> List[Dict[str, str]]:
        all_activities = []
        with SB(uc=True, headless=True, browser="chrome") as sb:
            for page in range(1, self.max_pages + 1):
                url = self.base_url if page == 1 else f"{self.base_url}/{page}"
                sb.open(url)
                sb.wait_for_element_present("body", timeout=5)
                html_content = sb.get_page_source()
                activities = self._parse_html(html_content)
                if not activities:
                    print(f"No activities found on page {page} for {self.base_url}. Stopping.")
                    break
                all_activities.extend(activities)
                print(f"Scraped {len(activities)} activities from page {page} of {self.base_url}")
                time.sleep(3.2 + random.uniform(0, 1))  # Randomize sleep between 3.2 and 4.2 seconds
        return all_activities

    def _parse_html(self, html_content: str) -> List[Dict[str, str]]:
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        activity_cards = soup.find_all("div", class_="productCardWrapper__BdSC")
        activities = []

        for card in activity_cards:
            title_tag = card.find("span", class_="title__h6oh")
            title = title_tag.find("strong").get_text(strip=True).replace('"', '') if title_tag else "Unknown Activity"

            rating_tag = card.find("div", class_="starRating__VZ9P")
            if rating_tag and "aria-label" in rating_tag.attrs:
                aria_label = rating_tag["aria-label"]
                parts = aria_label.split()
                rating_score = parts[1]
                ratings_count_raw = parts[-2].replace(",", "")
                ratings_count = int(ratings_count_raw) if ratings_count_raw.isdigit() else 0
            else:
                rating_score = "N/A"
                ratings_count = 0

            activities.append({
                "title": title,
                "ratings_count": str(ratings_count),
                "rating_score": rating_score
            })

        return activities


# --- Klook Scraper ---

class KlookScraper(ActivityScraper):
    def __init__(self, base_url: str, max_pages: int = Config.MAX_PAGES):
        self.base_url = base_url
        self.max_pages = max_pages

    def scrape(self) -> List[Dict[str, str]]:
        all_activities = []
        with SB(uc=True, headless=True, browser="chrome") as sb:
            for page in range(1, self.max_pages + 1):
                url = self.base_url if page == 1 else f"{self.base_url}?start={page}"
                sb.open(url)
                sb.wait_for_element_present("body", timeout=5)
                html_content = sb.get_page_source()
                activities = self._parse_html(html_content)
                if not activities:
                    print(f"No activities found on page {page} for {self.base_url}. Stopping.")
                    break
                all_activities.extend(activities)
                print(f"Scraped {len(activities)} activities from page {page} of {self.base_url}")
                time.sleep(3.2 + random.uniform(0, 1))  # Randomize sleep between 3.2 and 4.2 seconds
        return all_activities

    def _parse_html(self, html_content: str) -> List[Dict[str, str]]:
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        activity_cards = soup.find_all("div", class_="j_activity_item")
        activities = []

        for card in activity_cards:
            title_tag = card.find("a", class_="title")
            title = title_tag.get_text(strip=True).replace('"', '') if title_tag else "Unknown Activity"

            star_box = card.find("li", class_="star_box")
            rating_score = "N/A"
            ratings_count = 0

            if star_box:
                score_tag = star_box.find("span", class_="star_score")
                reviews_tag = star_box.find("span", class_="u_t_gray_9")

                if score_tag and reviews_tag:
                    rating_score = score_tag.get_text(strip=True)
                    reviews_text = reviews_tag.get_text(strip=True).strip("()").replace(" reviews", "").replace(",", "")
                    ratings_count = int(reviews_text) if reviews_text.isdigit() else 0

            if rating_score != "N/A" and ratings_count > 0:
                activities.append({
                    "title": title,
                    "ratings_count": str(ratings_count),
                    "rating_score": rating_score
                })

        return activities


# --- Utility Functions ---

def print_activities(activities: List[Dict[str, str]]) -> None:
    """Prints activity details in a formatted manner."""
    print("Activity Details:")
    for activity in activities:
        print(f"Title: {activity['title']}")
        print(f"Ratings Count: {activity['ratings_count']}")
        print(f"Rating Score: {activity['rating_score']}")
        print(f"Region: {activity['region']}")
        print(f"Source: {activity['source']}")
        print("-" * 50)


def save_to_csv(activities: List[Dict[str, str]], filename: str) -> None:
    """Saves activity details to a CSV file with 'source' and 'region' columns."""
    if not activities:
        print("No activities to save to CSV.")
        return

    headers = ["title", "ratings_count", "rating_score", "region", "source"]
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for activity in activities:
            writer.writerow(activity)
    print(f"Activity details saved to {filename}")


# --- Main Logic ---

def get_scraper(config: Dict[str, str]) -> Optional[ActivityScraper]:
    """Returns the appropriate scraper based on the source."""
    source = config["source"]
    url = config["url"]
    if source == "getyourguide":
        return GetYourGuideScraper(url)
    elif source == "viator":
        return ViatorScraper(url)
    elif source == "klook":
        return KlookScraper(url)
    else:
        print(f"Unknown source '{source}' for {config['region']}. Skipping.")
        return None


def scraper():
    """Executes the scraping process for all configured URLs, ensuring no duplicate titles."""
    all_activities = []
    seen_titles = set()  # Track unique titles

    for config in Config.URL_CONFIGS:
        scraper = get_scraper(config)
        if scraper:
            activities = scraper.scrape()
            for activity in activities:
                title = activity["title"]
                if title not in seen_titles:
                    seen_titles.add(title)
                    activity["region"] = config["region"]
                    activity["source"] = config["source"]
                    all_activities.append(activity)
                else:
                    print(f"Skipping duplicate title: '{title}' from {config['source']} ({config['region']})")

    print_activities(all_activities)
    save_to_csv(all_activities, Config.OUTPUT_CSV_FILE)


scraper()
