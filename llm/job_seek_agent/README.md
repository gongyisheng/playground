# Job Analysis System

A comprehensive job analysis system that crawls LinkedIn job postings and analyzes them against candidate experience using AI. The system includes a modern React.js frontend for easy interaction and a FastAPI backend for processing.

## Features

- **LinkedIn Job Crawling**: Crawl job postings with advanced filtering options
- **AI-Powered Analysis**: Analyze job matches using OpenAI GPT-4
- **Real-time Progress Tracking**: Monitor job analysis progress in real-time
- **Modern React Interface**: Beautiful, responsive frontend built with React.js
- **Comprehensive Filtering**: Filter by experience level, job type, work type, and more
- **Skill Matching**: Identify matched and missing skills for each job

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Job Analysis  │
│   (React.js)    │◄──►│   (FastAPI)     │◄──►│   (Python)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   LinkedIn      │
                       │   API           │
                       └─────────────────┘
```

## Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- OpenAI API key
- Internet connection for LinkedIn crawling

## Installation

1. **Clone the repository**:
   ```bash
   cd llm/job_seek_agent
   ```

2. **Install backend dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

4. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

5. **Prepare candidate experience file**:
   - Ensure `candidate_experience.test.txt` contains your experience
   - Or update the path in the backend configuration

## Usage

### Starting the Backend

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Start the FastAPI server**:
   ```bash
   python main.py
   ```

   The server will start on `http://localhost:8000`

3. **Optional: Start with uvicorn directly**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Starting the Frontend

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Start the React development server**:
   ```bash
   npm start
   ```

   The frontend will start on `http://localhost:3000`

3. **Build for production** (optional):
   ```bash
   npm run build
   ```

### Using the Application

1. **Open your browser** and navigate to `http://localhost:3000`

2. **Configure job search parameters**:
   - **Keywords**: Job title or skills to search for
   - **Location**: Geographic location for job search
   - **Number of Jobs**: How many jobs to analyze (1-100)
   - **Experience Level**: Filter by experience requirements
   - **Job Type**: Full-time, part-time, contract, etc.
   - **Work Type**: On-site, remote, or hybrid
   - **Time Posted**: Recent job postings filter

3. **Start analysis**:
   - Click "Start Analysis" to begin the job analysis
   - Monitor progress in real-time
   - View results when complete

### API Endpoints

The backend provides the following REST API endpoints:

- `GET /` - API status
- `GET /health` - Health check
- `POST /jobs/start` - Start a new job analysis
- `GET /jobs/{job_id}` - Get job status and results
- `GET /jobs` - List all jobs

### Example API Usage

```bash
# Start a job analysis
curl -X POST "http://localhost:8000/jobs/start" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": "Python developer",
    "location": "San Francisco, CA",
    "limit": 10,
    "f_E": 4,
    "f_JT": "F",
    "f_WT": 2
  }'

# Check job status
curl "http://localhost:8000/jobs/job_20231201_143022"
```

## Configuration

### Backend Configuration

The backend can be configured by modifying the following:

- **Candidate Experience File**: Update the path in `main.py`
- **API Port**: Change the port in `main.py` (default: 8000)
- **CORS Settings**: Update CORS origins for production

### Frontend Configuration

- **API URL**: Update `REACT_APP_API_URL` environment variable or modify `App.js`
- **Styling**: Modify CSS in `src/index.css` for custom appearance
- **Components**: Extend React components in `src/components/` for new features

## Project Structure

```
llm/job_seek_agent/
├── backend/
│   ├── main.py              # FastAPI backend server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── public/
│   │   ├── index.html       # Main HTML template
│   │   └── manifest.json    # PWA manifest
│   ├── src/
│   │   ├── components/
│   │   │   ├── JobForm.js   # Job search form component
│   │   │   ├── JobStatus.js # Job status display component
│   │   │   └── JobResults.js # Results display component
│   │   ├── App.js           # Main React application
│   │   ├── index.js         # React entry point
│   │   └── index.css        # Global styles
│   └── package.json         # Node.js dependencies
├── agent.py                # AI job analysis logic
├── crawler.py              # LinkedIn job crawler
├── cronjob.py              # Job analysis orchestration
├── candidate_experience.test.txt  # Candidate experience data
└── results/                # Analysis output directory
```

## Development

### Frontend Development

The React frontend is built with modern JavaScript and includes:

- **Component-based architecture** for maintainable code
- **State management** with React hooks
- **Real-time updates** with polling
- **Responsive design** for mobile and desktop
- **Error handling** with user-friendly messages

### Adding New Features

1. **Backend Extensions**:
   - Add new endpoints in `main.py`
   - Extend the `JobRequest` model for new parameters
   - Modify the analysis logic in `agent.py`

2. **Frontend Enhancements**:
   - Create new components in `src/components/`
   - Update the form in `JobForm.js`
   - Add new UI components and styling
   - Extend the JavaScript functionality

3. **Crawler Improvements**:
   - Add new LinkedIn API parameters in `crawler.py`
   - Implement additional job sources
   - Enhance job detail extraction

## Job Analysis Process

1. **Job Crawling**: 
   - Fetches job postings from LinkedIn using the guest API
   - Applies user-specified filters
   - Extracts job details and descriptions

2. **AI Analysis**:
   - Uses OpenAI GPT-4 to analyze each job posting
   - Extracts company name, position, and experience requirements
   - Compares job requirements with candidate experience
   - Identifies matched and missing skills

3. **Results Processing**:
   - Saves results to JSON files
   - Provides real-time progress updates
   - Displays results in the React interface

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Set**:
   - Ensure `OPENAI_API_KEY` environment variable is set
   - Check that the API key is valid and has sufficient credits

2. **LinkedIn Rate Limiting**:
   - The crawler includes delays between requests
   - If you encounter rate limiting, increase the delay in `crawler.py`

3. **CORS Issues**:
   - Ensure the frontend and backend are running on the correct ports
   - Check CORS configuration in the backend

4. **React Build Issues**:
   - Ensure Node.js and npm are properly installed
   - Clear npm cache: `npm cache clean --force`
   - Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

5. **Job Analysis Failures**:
   - Check the candidate experience file exists and is readable
   - Verify internet connectivity for LinkedIn API access

### Logs

The system provides comprehensive logging:
- Backend logs are displayed in the terminal
- Frontend errors are shown in the browser console
- Job analysis progress is tracked in real-time

## Production Deployment

### Backend Deployment

1. **Using Docker** (recommended):
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY backend/requirements.txt .
   RUN pip install -r requirements.txt
   COPY backend/ .
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Using a cloud platform**:
   - Deploy to Heroku, Railway, or similar platforms
   - Set environment variables for API keys
   - Configure CORS for your domain

### Frontend Deployment

1. **Build the production version**:
   ```bash
   cd frontend
   npm run build
   ```

2. **Deploy to a static hosting service**:
   - Netlify, Vercel, or GitHub Pages
   - Configure environment variables for API URL
   - Update CORS settings in backend

## License

This project is for educational and personal use. Please respect LinkedIn's terms of service when using the crawler.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Ensure all dependencies are installed correctly
4. Verify API keys and network connectivity 