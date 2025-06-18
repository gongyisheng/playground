import aiohttp
from typing import Dict, Optional, List, Tuple, Set
import logging
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LinkedInJobCrawler:
    def __init__(self):
        self.job_list_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        self.job_detail_url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/%s"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.delay = 2  # Delay between requests in seconds
        self.filtered_texts = {"Show more", "Show less", "See who you know"}
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_jobs_list(self) -> Optional[str]:
        """
        Fetch jobs from LinkedIn API
        Returns:
            Optional[str]: Response HTML if successful, None if failed
        """
        params = {
            "keywords": "software engineer",
            "location": "San Francisco, CA",
            "start": 0
        }
        try:
            logging.info("Fetching jobs from LinkedIn API")
            async with self.session.get(self.job_list_url, params=params, timeout=30) as response:
                response.raise_for_status()
                data = await response.text()
                logging.info(f"Successfully fetched jobs")
                return data
        except Exception as e:
            logging.error(f"Request failed: {str(e)}", exc_info=True)
            return None
    
    def parse_jobs_list(self, html_data: str) -> Dict[str, str]:
        """
        Parse HTML response to extract job links
        Args:
            html_data: Raw HTML response
        Returns:
            Dict[str, str]: Dictionary mapping job IDs to job URLs
        """
        if not html_data:
            return {}
            
        try:
            soup = BeautifulSoup(html_data, 'html.parser')
            job_links = []
            
            # Find all list items
            list_items = soup.find_all('li')
            logging.info(f"Found {len(list_items)} list items")
            
            # Extract links from each list item
            for item in list_items:
                # Find all anchor tags in the list item
                links = item.find_all('a')
                for link in links:
                    href = link.get('href')
                    if href and '/jobs/view/' in href:
                        # Convert relative URLs to absolute URLs
                        full_url = urljoin("https://www.linkedin.com", href)
                        job_links.append(full_url)
            
            logging.info(f"Extracted {len(job_links)} job links")

            job_map = {}
            for link in job_links:
                tmp = link.split("?")[0]
                job_id = tmp.split("-")[-1]
                job_map[job_id] = link
            return job_map
            
        except Exception as e:
            logging.error(f"Failed to parse HTML: {str(e)}", exc_info=True)
            return {}
    
    async def fetch_job_details(self, job_id: str) -> Optional[str]:
        """
        Fetch job details from LinkedIn API
        Returns:
            Optional[str]: Response HTML if successful, None if failed
        """
        try:
            logging.info(f"Fetching job details for {job_id}")
            async with self.session.get(self.job_detail_url % job_id, timeout=30) as response:
                response.raise_for_status()
                data = await response.text()
                logging.info(f"Successfully fetched job details for {job_id}")
                return data
        except Exception as e:
            logging.error(f"Request failed: {str(e)}", exc_info=True)
            return None
    
    def _extract_text_with_hierarchy(self, element: BeautifulSoup, level: int = 0, seen_text: Set[str] = None) -> List[Tuple[str, int]]:
        """
        Recursively extract text from elements while preserving hierarchy and removing duplicates
        Args:
            element: BeautifulSoup element
            level: Current nesting level
            seen_text: Set of already seen text entries
        Returns:
            List of tuples containing (text, level)
        """
        if seen_text is None:
            seen_text = set()
            
        result = []
        
        # If element has text directly
        if element.string and element.string.strip():
            text = element.string.strip()
            if text not in seen_text and text not in self.filtered_texts:
                seen_text.add(text)
                result.append((text, level))
        
        # Process child elements
        for child in element.children:
            if child.name:  # If it's a tag (not just text)
                # Add text from this element
                if child.string and child.string.strip():
                    text = child.string.strip()
                    if text not in seen_text and text not in self.filtered_texts:
                        seen_text.add(text)
                        result.append((text, level))
                # Recursively process children
                result.extend(self._extract_text_with_hierarchy(child, level + 1, seen_text))
            elif child.string and child.string.strip():  # If it's just text
                text = child.string.strip()
                if text not in seen_text and text not in self.filtered_texts:
                    seen_text.add(text)
                    result.append((text, level))
        
        return result
    
    def parse_job_details(self, data: str) -> Optional[str]:
        """
        Parse HTML response to extract job details with hierarchy
        Args:
            data: Raw HTML response
        Returns:
            Optional[str]: Formatted job details text
        """
        if not data:
            return None
            
        try:
            soup = BeautifulSoup(data, 'html.parser')
            
            # Find the main job details section
            details_section = soup.find('div', class_='decorated-job-posting__details')
            if not details_section:
                logging.warning("Could not find job details section")
                return None
            
            # Extract text with hierarchy
            text_content = self._extract_text_with_hierarchy(details_section)
            
            logging.info(f"Extracted {len(text_content)} unique text elements from job details")

            document = ""             
            document += "Details:\n"
            for text, level in text_content:
                indent = "  " * level  # 2 spaces per level
                document += f"{indent}- {text}\n"
            return document
            
        except Exception as e:
            logging.error(f"Failed to parse job details: {str(e)}", exc_info=True)
            return None

async def run_crawler():
    async with LinkedInJobCrawler() as crawler:
        # Fetch job listings
        jobs_list_html = await crawler.fetch_jobs_list()
        job_map = crawler.parse_jobs_list(jobs_list_html)

        job_details = []
        for job_id, job_link in job_map.items():
            # Add delay between requests
            await asyncio.sleep(crawler.delay)
            
            job_detail_html = await crawler.fetch_job_details(job_id)
            job_detail_texts = crawler.parse_job_details(job_detail_html)
            job_details.append({
                "job_id": job_id,
                "job_link": job_link,
                "job_detail_texts": job_detail_texts
            })
        return job_details

if __name__ == "__main__":
    import json
    job_details = asyncio.run(run_crawler())
    with open("job_details.json", "w") as f:
        json.dump(job_details, f)