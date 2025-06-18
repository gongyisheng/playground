import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from crawler import run_crawler
from agent import run_analyze_job_posting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JobAnalysisJob:
    def __init__(self, 
                 config_path: str = "candidate_experience.test.txt",
                 keywords: str = "software engineer",
                 location: str = "San Francisco, CA",
                 limit: int = 10,
                 f_AL: bool = False,
                 f_E: Optional[int] = None,
                 f_JT: Optional[str] = None,
                 f_WT: Optional[int] = None,
                 f_JIYN: bool = False,
                 f_PP: Optional[str] = None,
                 f_C: Optional[str] = None,
                 f_TPR: Optional[str] = None):
        """
        Initialize the job analysis job with configuration and crawler parameters
        Args:
            config_path: Path to the candidate experience text file
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
        """
        self.config_path = Path(config_path)
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Crawler parameters
        self.keywords = keywords
        self.location = location
        self.limit = limit
        self.f_AL = f_AL
        self.f_E = f_E
        self.f_JT = f_JT
        self.f_WT = f_WT
        self.f_JIYN = f_JIYN
        self.f_PP = f_PP
        self.f_C = f_C
        self.f_TPR = f_TPR

    def load_candidate_experience(self) -> str:
        """
        Load candidate experience from text file
        Returns:
            str: Candidate experience text
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Candidate experience file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                candidate_experience = f.read()
            
            if not candidate_experience.strip():
                raise ValueError(f"Candidate experience file is empty: {self.config_path}")
            
            return candidate_experience
            
        except Exception as e:
            raise Exception(f"Error loading candidate experience: {str(e)}")

    async def analyze_jobs(self):
        """
        Run job analysis workflow
        """
        try:
            # Load candidate experience
            candidate_experience = self.load_candidate_experience()
            
            logging.info(f"Starting job crawler with keywords: {self.keywords}, location: {self.location}, limit: {self.limit}")
            cutoff_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            part_num = 0
            start = 0
            while start < self.limit:
                job_details = await run_crawler(
                    keywords=self.keywords,
                    location=self.location,
                    start=start,
                    f_AL=self.f_AL,
                    f_E=self.f_E,
                    f_JT=self.f_JT,
                    f_WT=self.f_WT,
                    f_JIYN=self.f_JIYN,
                    f_PP=self.f_PP,
                    f_C=self.f_C,
                    f_TPR=self.f_TPR
                )
                start += len(job_details)

                analyze_results = []
                for item in job_details:
                    result = await run_analyze_job_posting(item["job_detail_texts"], candidate_experience)
                    analyze_results.append({
                        "job_id": item["job_id"],
                        "job_link": item["job_link"],
                        "company_name": result.company_name,
                        "position_name": result.position_name,
                        "min_years_experience": result.min_years_experience,
                        "is_match": result.is_match,
                        "matched_skills": result.matched_skills,
                        "miss_skills": result.miss_skills
                    })
                # Save results to file
                output_file = self.output_dir / f"job_analysis_{cutoff_str}_part_{part_num}.json"
                with open(output_file, 'w') as f:
                    json.dump(analyze_results, f, indent=4)
                
                logging.info(f"Analysis completed. Results saved to {output_file}")
                logging.info(f"Crawled and analyzed {start + len(analyze_results)} out of {self.limit} jobs successfully")
                part_num += 1
            
        except Exception as e:
            logging.error(f"Error in job analysis: {str(e)}", exc_info=True)
            raise

async def main():
    """
    Main function to run the job analysis job
    """
    # Example: Search for Python developer jobs in New York with specific filters
    job_analysis = JobAnalysisJob(
        config_path="candidate_experience.test.txt",
        keywords="Software Engineer",
        location="San Francisco Bay Area",
        limit=25,
        f_E=3,  # Mid-Senior level
        f_JT="F",  # Full-time only
        f_TPR="r604800"  # Past week
    )
    
    # Or use default settings
    # job_analysis = JobAnalysisJob()
    
    await job_analysis.analyze_jobs()

if __name__ == "__main__":
    asyncio.run(main()) 