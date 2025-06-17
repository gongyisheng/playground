from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import openai
import yaml
import logging

# Load environment variables
load_dotenv()

# Load YAML configuration
def load_config():
    with open("backend-config.test.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        # Replace environment variables in config
        for key, value in config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str) and subvalue.startswith("${") and subvalue.endswith("}"):
                        env_var = subvalue[2:-1]
                        value[subkey] = os.getenv(env_var)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var)
        return config

config = load_config()

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Job Seek Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["cors"]["allow_origins"],
    allow_credentials=config["cors"]["allow_credentials"],
    allow_methods=config["cors"]["allow_methods"],
    allow_headers=config["cors"]["allow_headers"],
)

# Models
class JobSearchParams(BaseModel):
    keyword: str
    time_range: str  # e.g., "24h", "7d", "30d"
    location: Optional[str] = None

class UserProfile(BaseModel):
    years_of_experience: int
    skills: List[str]

class JobPost(BaseModel):
    title: str
    company: str
    description: str
    requirements: str
    location: str
    posted_date: str
    url: str

# Initialize OpenAI
openai.api_key = config["openai_api_key"]

def get_linkedin_access_token():
    """Get LinkedIn access token using client credentials"""
    url = config["linkedin"]["auth_url"]
    data = {
        "grant_type": "client_credentials",
        "client_id": config["linkedin"]["client_id"],
        "client_secret": config["linkedin"]["client_secret"]
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        logging.error(f"Failed to get LinkedIn access token: {response.status_code} {response.text}")
        raise HTTPException(status_code=401, detail="Failed to get LinkedIn access token")

@app.post("/search-jobs")
async def search_jobs(params: JobSearchParams):
    """Search for jobs on LinkedIn based on keywords and time range"""
    try:
        access_token = get_linkedin_access_token()
        
        # Calculate time range
        now = datetime.now()
        if params.time_range == "24h":
            time_delta = timedelta(days=1)
        elif params.time_range == "7d":
            time_delta = timedelta(days=7)
        elif params.time_range == "30d":
            time_delta = timedelta(days=30)
        else:
            raise HTTPException(status_code=400, detail="Invalid time range")

        # LinkedIn API endpoint for job search
        url = f"{config['linkedin']['api_base_url']}/jobs"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare search parameters
        search_params = {
            "keywords": params.keyword,
            "postedAfter": (now - time_delta).isoformat(),
            "location": params.location if params.location else None
        }
        
        response = requests.get(url, headers=headers, params=search_params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch jobs from LinkedIn")
            
    except Exception as e:
        logging.error(f"Error in search_jobs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-job-match")
async def analyze_job_match(job: JobPost, user_profile: UserProfile):
    """Analyze if a job matches the user's profile using AI"""
    try:
        # Prepare the prompt for OpenAI
        prompt = f"""
        Analyze if this job is a good match for the candidate based on their profile.
        
        Job Title: {job.title}
        Company: {job.company}
        Description: {job.description}
        Requirements: {job.requirements}
        
        Candidate Profile:
        Years of Experience: {user_profile.years_of_experience}
        Skills: {', '.join(user_profile.skills)}
        
        Please analyze and provide:
        1. Match score (0-100)
        2. Key matching points
        3. Potential gaps
        4. Overall recommendation
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a job matching expert."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "analysis": response.choices[0].message.content,
            "job_details": job.dict(),
            "user_profile": user_profile.dict()
        }
        
    except Exception as e:
        logging.error(f"Error in analyze_job_match: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config["server"]["host"], 
        port=config["server"]["port"]
    )
