# Job Seek Agent

An AI-powered job search application that helps you find the right jobs on LinkedIn based on your skills and experience.

## Features

1. Search jobs from LinkedIn by keyword and time range
2. AI-powered job matching based on your skills and experience
3. Detailed analysis of job compatibility

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with the following variables:
```
LINKEDIN_CLIENT_ID=your_linkedin_client_id
LINKEDIN_CLIENT_SECRET=your_linkedin_client_secret
OPENAI_API_KEY=your_openai_api_key
```

3. Get your API keys:
   - LinkedIn API: Create a LinkedIn Developer account and create a new application to get your client ID and secret
   - OpenAI API: Sign up at https://platform.openai.com to get your API key

## Running the Application

Start the server:
```bash
python server.py
```

The server will run on http://localhost:8000

## API Endpoints

### 1. Search Jobs
- Endpoint: POST `/search-jobs`
- Body:
```json
{
    "keyword": "python developer",
    "time_range": "7d",
    "location": "San Francisco"
}
```

### 2. Analyze Job Match
- Endpoint: POST `/analyze-job-match`
- Body:
```json
{
    "job": {
        "title": "Senior Python Developer",
        "company": "Tech Corp",
        "description": "...",
        "requirements": "...",
        "location": "San Francisco",
        "posted_date": "2024-01-01",
        "url": "https://linkedin.com/jobs/..."
    },
    "user_profile": {
        "years_of_experience": 5,
        "skills": ["Python", "Django", "AWS"]
    }
}
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and rotate them periodically
- Use environment variables in production 