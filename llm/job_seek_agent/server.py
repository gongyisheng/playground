from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import openai

# Load environment variables
load_dotenv()

app = FastAPI(title="Job Seek Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# LinkedIn API configuration
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

def get_linkedin_access_token():
    """Get LinkedIn access token using client credentials"""
    url = "https://www.linkedin.com/oauth/v2/accessToken"
    data = {
        "grant_type": "client_credentials",
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
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
        url = "https://api.linkedin.com/v2/jobs"
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
            model="gpt-3.5-turbo",
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
