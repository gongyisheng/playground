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

    async def fetch_jobs_list(self, 
                             keywords: str = "software engineer",
                             location: str = "San Francisco, CA",
                             start: int = 0,
                             f_AL: bool = False,
                             f_E: Optional[int] = None,
                             f_JT: Optional[str] = None,
                             f_WT: Optional[int] = None,
                             f_JIYN: bool = False,
                             f_PP: Optional[str] = None,
                             f_C: Optional[str] = None,
                             f_TPR: Optional[str] = None) -> Optional[str]:
        """
        Fetch jobs from LinkedIn API with comprehensive filtering options
        
        Args:
            keywords: A word or phrase used as the main filter in the search
            location: A country to narrow down the job search (use English terms, for Brazil use "Brazil")
            start: Sets the starting job position for pagination (works in increments of 25: 0, 25, 50, etc.)
            f_AL: Filter for jobs with simplified application (True to activate)
            f_E: Filter by Experience Level (1=Intern, 2=Assistant, 3=Junior, 4=Mid-Senior, 5=Director, 6=Executive)
            f_JT: Filter by Job Type (F=Full-time, P=Part-time, C=Contract, T=Temporary, V=Volunteer, I=Internship, O=Other)
            f_WT: Filter by Work Schedule Model (1=On-site, 2=Remote, 3=Hybrid)
            f_JIYN: Filter for jobs with fewer than 10 applicants (True to activate)
            f_PP: Filter for jobs in a specific City (requires city ID)
            f_C: Filter for jobs from a specific Company (requires company ID)
            f_TPR: Filter by time posted (""=any time, "r86400"=past 24h, "r604800"=past week, "r2592000"=past month)
            
        Returns:
            Optional[str]: Response HTML if successful, None if failed
        """
        params = {
            "keywords": keywords,
            "location": location,
            "start": start
        }
        
        # Add optional filters only if they are provided
        if f_AL:
            params["f_AL"] = "true"
        
        if f_E is not None:
            params["f_E"] = str(f_E)
            
        if f_JT is not None:
            params["f_JT"] = f_JT
            
        if f_WT is not None:
            params["f_WT"] = str(f_WT)
            
        if f_JIYN:
            params["f_JIYN"] = "true"
            
        if f_PP is not None:
            params["f_PP"] = f_PP
            
        if f_C is not None:
            params["f_C"] = f_C
            
        if f_TPR is not None:
            params["f_TPR"] = f_TPR
        
        try:
            logging.info(f"Fetching jobs from LinkedIn API with params: {params}")
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

async def run_crawler(keywords: str = "software engineer",
                     location: str = "San Francisco, CA",
                     start: int = 0,
                     f_AL: bool = False,
                     f_E: Optional[int] = None,
                     f_JT: Optional[str] = None,
                     f_WT: Optional[int] = None,
                     f_JIYN: bool = False,
                     f_PP: Optional[str] = None,
                     f_C: Optional[str] = None,
                     f_TPR: Optional[str] = None):
    """
    Run the LinkedIn job crawler with customizable search parameters
    
    Args:
        keywords: A word or phrase used as the main filter in the search
        location: A country to narrow down the job search
        start: Sets the starting job position for pagination
        f_AL: Filter for jobs with simplified application
        f_E: Filter by Experience Level (1=Intern, 2=Assistant, 3=Junior, 4=Mid-Senior, 5=Director, 6=Executive)
        f_JT: Filter by Job Type (F=Full-time, P=Part-time, C=Contract, T=Temporary, V=Volunteer, I=Internship, O=Other)
        f_WT: Filter by Work Schedule Model (1=On-site, 2=Remote, 3=Hybrid)
        f_JIYN: Filter for jobs with fewer than 10 applicants
        f_PP: Filter for jobs in a specific City (requires city ID)
        f_C: Filter for jobs from a specific Company (requires company ID)
        f_TPR: Filter by time posted (""=any time, "r86400"=past 24h, "r604800"=past week, "r2592000"=past month)
        
    Returns:
        List of job details
    """
    async with LinkedInJobCrawler() as crawler:
        # Fetch job listings with custom parameters
        jobs_list_html = await crawler.fetch_jobs_list(
            keywords=keywords,
            location=location,
            start=start,
            f_AL=f_AL,
            f_E=f_E,
            f_JT=f_JT,
            f_WT=f_WT,
            f_JIYN=f_JIYN,
            f_PP=f_PP,
            f_C=f_C,
            f_TPR=f_TPR
        )
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