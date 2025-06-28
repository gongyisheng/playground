from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import logging
import json
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JobMatchResult(BaseModel):
    """Schema for job match result"""
    company_name: str = Field(description="Company name extracted from the job posting")
    position_name: str = Field(description="Position/title extracted from the job posting")
    min_years_experience: str = Field(description="Minimum years of experience requirement extracted from the job posting")
    is_match: bool = Field(description="Whether the job matches the candidate's experience and skills")
    matched_skills: List[str] = Field(description="List of skills that the job matches the candidate's experience and skills")
    miss_skills: List[str] = Field(description="List of skills mentioned in the job posting that don't match the candidate's experience")
    # matched_experience: List[str] = Field(description="List of experiences that the job matches the candidate's experience and skills")

class JobAnalyzer:
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the job analyzer with OpenAI API key
        Args:
            openai_api_key: OpenAI API key. If not provided, will try to get from environment variable
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4.1-2025-04-14",
            temperature=0,
            openai_api_key=self.api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=JobMatchResult)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a job matching expert. Your task is to analyze a job posting and determine if it matches a candidate's experience and skills.

            First, extract the following information from the job posting:
            - Company name: The name of the company posting the job
            - Position name: The job title/position being offered
            - Minimum years of experience: The required years of experience mentioned in the posting

            Then, analyze the match by considering:
            1. Required skills and experience in the job posting
            2. Candidate's past experience and skills
            3. Industry relevance
            4. Technical requirements
            5. Years of experience requirement
            
            For matched skills:
            - List only the specific skills that directly match between the job requirements and candidate experience
            - Keep skill names short and precise (e.g., "Python" instead of "Python programming language")
            - Focus on technical skills, tools, and technologies rather than general descriptions
            
            For miss_skills:
            - List skills mentioned in the job posting that the candidate doesn't have experience with
            - Keep skill names short and precise (e.g., "React" instead of "React.js framework")
            - Focus on technical skills, tools, and technologies that are clearly required but missing from candidate's background
            
            Output the extracted information (company name, position name, minimum years of experience), whether the job is a match (yes/no), matched skills, and miss_skills.
            {format_instructions}"""),
            ("human", """Job Description:
            {job_description}
            
            Candidate Experience:
            {candidate_experience}""")
        ])

    async def analyze_job_match(self, job_description: str, candidate_experience: str) -> JobMatchResult:
        """
        Analyze if a job posting matches the candidate's experience and skills
        Args:
            job_description: The job posting description
            candidate_experience: Candidate's work experience
        Returns:
            JobMatchResult containing extracted job info, match decision and skills
        """
        try:
            # Format the prompt
            prompt = self.prompt_template.format_messages(
                job_description=job_description,
                candidate_experience=candidate_experience,
                format_instructions=self.parser.get_format_instructions()
            )
            
            # Get response from LLM
            response = await self.llm.ainvoke(prompt)
            
            # Parse the response
            result = self.parser.parse(response.content)
            
            logging.info(f"Job match analysis completed. Match: {result.is_match}")
            return result
            
        except Exception as e:
            logging.error(f"Error analyzing job match: {str(e)}", exc_info=True)
            raise

async def run_analyze_job_posting(job_description: str, candidate_experience: str) -> JobMatchResult:
    """
    Convenience function to analyze a job posting
    Args:
        job_description: The job posting description
        candidate_experience: Candidate's work experience
    Returns:
        JobMatchResult: Analysis result containing extracted job info, match decision and skills
    """
    try:
        analyzer = JobAnalyzer()
        result = await analyzer.analyze_job_match(
            job_description=job_description,
            candidate_experience=candidate_experience,
        )
        return result
    except Exception as e:
        logging.error(f"Error in job posting analysis: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    job_details = json.load(open("job_details.json"))
    candidate_experience = open("candidate_experience.test.txt").read()
    for job in job_details:
        result = asyncio.run(run_analyze_job_posting(job["job_detail_texts"], candidate_experience))
        print(result)