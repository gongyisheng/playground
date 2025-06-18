from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cronjob import JobAnalysisJob
from agent import JobMatchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="Job Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store job status
job_status = {}

class JobRequest(BaseModel):
    keywords: str = "software engineer"
    location: str = "San Francisco, CA"
    limit: int = 10
    f_AL: bool = False
    f_E: Optional[int] = None
    f_JT: Optional[str] = None
    f_WT: Optional[int] = None
    f_JIYN: bool = False
    f_PP: Optional[str] = None
    f_C: Optional[str] = None
    f_TPR: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    total: int = 0
    message: str = ""
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Job Analysis API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/jobs/start", response_model=JobStatus)
async def start_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Start a new job analysis task"""
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize job status
    job_status[job_id] = {
        "status": "running",
        "progress": 0,
        "total": request.limit,
        "message": "Job started",
        "results": [],
        "error": None
    }
    
    # Add the job to background tasks
    background_tasks.add_task(run_job_analysis, job_id, request)
    
    return JobStatus(
        job_id=job_id,
        status="running",
        progress=0,
        total=request.limit,
        message="Job started successfully"
    )

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    return JobStatus(
        job_id=job_id,
        status=status["status"],
        progress=status["progress"],
        total=status["total"],
        message=status["message"],
        results=status["results"],
        error=status["error"]
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": status["status"],
                "progress": status["progress"],
                "total": status["total"],
                "message": status["message"]
            }
            for job_id, status in job_status.items()
        ]
    }

async def run_job_analysis(job_id: str, request: JobRequest):
    """Run the job analysis in the background"""
    try:
        logging.info(f"Starting job analysis for job_id: {job_id}")
        
        # Update status
        job_status[job_id]["message"] = "Initializing job analysis..."
        
        # Create job analysis instance
        job_analysis = JobAnalysisJob(
            config_path="candidate_experience.test.txt",
            keywords=request.keywords,
            location=request.location,
            limit=request.limit,
            f_AL=request.f_AL,
            f_E=request.f_E,
            f_JT=request.f_JT,
            f_WT=request.f_WT,
            f_JIYN=request.f_JIYN,
            f_PP=request.f_PP,
            f_C=request.f_C,
            f_TPR=request.f_TPR
        )
        
        # Load candidate experience
        job_status[job_id]["message"] = "Loading candidate experience..."
        candidate_experience = job_analysis.load_candidate_experience()
        
        # Run the analysis
        job_status[job_id]["message"] = "Starting job crawler..."
        cutoff_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        part_num = 0
        start = 0
        all_results = []
        
        while start < request.limit:
            job_status[job_id]["message"] = f"Crawling jobs (part {part_num + 1})..."
            
            job_details = await job_analysis.run_crawler_batch(
                keywords=request.keywords,
                location=request.location,
                start=start,
                f_AL=request.f_AL,
                f_E=request.f_E,
                f_JT=request.f_JT,
                f_WT=request.f_WT,
                f_JIYN=request.f_JIYN,
                f_PP=request.f_PP,
                f_C=request.f_C,
                f_TPR=request.f_TPR
            )
            
            if not job_details:
                break
                
            start += len(job_details)
            
            # Analyze jobs
            job_status[job_id]["message"] = f"Analyzing {len(job_details)} jobs..."
            analyze_results = []
            
            for i, item in enumerate(job_details):
                try:
                    result = await job_analysis.run_analyze_job_posting(
                        item["job_detail_texts"], 
                        candidate_experience
                    )
                    
                    if result:
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
                    
                    # Update progress
                    job_status[job_id]["progress"] = len(all_results) + len(analyze_results)
                    
                except Exception as e:
                    logging.error(f"Error analyzing job {item['job_id']}: {str(e)}")
                    continue
            
            all_results.extend(analyze_results)
            
            # Save partial results
            output_file = job_analysis.output_dir / f"job_analysis_{cutoff_str}_part_{part_num}.json"
            with open(output_file, 'w') as f:
                json.dump(analyze_results, f, indent=4)
            
            part_num += 1
            
            # Check if we've reached the limit
            if len(all_results) >= request.limit:
                all_results = all_results[:request.limit]
                break
        
        # Update final status
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["progress"] = len(all_results)
        job_status[job_id]["message"] = f"Analysis completed. Found {len(all_results)} jobs."
        job_status[job_id]["results"] = all_results
        
        logging.info(f"Job analysis completed for job_id: {job_id}")
        
    except Exception as e:
        logging.error(f"Error in job analysis for job_id {job_id}: {str(e)}", exc_info=True)
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        job_status[job_id]["message"] = f"Job failed: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 