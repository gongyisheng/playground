import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from crawler import run_crawler
from agent import run_analyze_job_posting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JobAnalysisCronjob:
    def __init__(self, config_path: str = "candidate_config.json"):
        """
        Initialize the cronjob with configuration file path
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_path = Path(config_path)
        self.output_dir = Path("job_analysis_results")
        self.output_dir.mkdir(exist_ok=True)

    def load_candidate_config(self) -> dict:
        """
        Load candidate configuration from JSON file
        Returns:
            dict: Candidate configuration
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ["candidate_experience", "candidate_skills"]
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in config: {', '.join(missing_fields)}")
            
            return config
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {self.config_path}")
        except Exception as e:
            raise Exception(f"Error loading configuration: {str(e)}")

    async def analyze_jobs(self):
        """
        Run job analysis workflow
        """
        try:
            # Load candidate configuration
            config = self.load_candidate_config()
            candidate_experience = config["candidate_experience"]
            candidate_skills = config["candidate_skills"]
            
            logging.info("Starting job crawler")
            job_details = await run_crawler()

            analyze_results = []
            for item in job_details:
                result = await run_analyze_job_posting(item["job_detail_texts"], candidate_experience, candidate_skills)
                analyze_results.append({
                    "job_id": item["job_id"],
                    "job_link": item["job_link"],
                    "is_match": result.is_match,
                    "matched_skills": result.matched_skills
                })
            
            # Save results to file
            output_file = self.output_dir / f"job_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(analyze_results, f, indent=4)
            
            logging.info(f"Analysis completed. Results saved to {output_file}")
            logging.info(f"Crawled and analyzed {len(analyze_results)} jobs successfully")
            
        except Exception as e:
            logging.error(f"Error in job analysis: {str(e)}", exc_info=True)
            raise

async def main():
    """
    Main function to run the cronjob
    """
    cronjob = JobAnalysisCronjob()
    await cronjob.analyze_jobs()

if __name__ == "__main__":
    asyncio.run(main()) 