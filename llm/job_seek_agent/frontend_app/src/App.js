import React, { useState, useEffect } from 'react';
import axios from 'axios';
import JobForm from './components/JobForm';
import JobStatus from './components/JobStatus';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [currentJob, setCurrentJob] = useState(null);
  const [statusInterval, setStatusInterval] = useState(null);

  useEffect(() => {
    // Check API health on component mount
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await axios.get(`${API_BASE_URL}/health`);
      console.log('API is healthy');
    } catch (error) {
      console.error('API health check failed:', error);
    }
  };

  const startJob = async (requestData) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/jobs/start`, requestData);
      const result = response.data;
      
      setCurrentJob(result);
      startStatusPolling(result.job_id);
      
    } catch (error) {
      console.error('Error starting job:', error);
      alert('Error starting job analysis: ' + (error.response?.data?.detail || error.message));
    }
  };

  const startStatusPolling = (jobId) => {
    if (statusInterval) {
      clearInterval(statusInterval);
    }
    
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/jobs/${jobId}`);
        const status = response.data;
        
        setCurrentJob(status);
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          setStatusInterval(null);
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    }, 2000);
    
    setStatusInterval(interval);
  };

  const handleFormSubmit = (formData) => {
    const requestData = {
      keywords: formData.keywords,
      location: formData.location,
      limit: parseInt(formData.limit),
      f_AL: formData.f_AL,
      f_E: formData.f_E ? parseInt(formData.f_E) : null,
      f_JT: formData.f_JT || null,
      f_WT: formData.f_WT ? parseInt(formData.f_WT) : null,
      f_JIYN: formData.f_JIYN,
      f_PP: null,
      f_C: null,
      f_TPR: formData.f_TPR || null
    };

    startJob(requestData);
  };

  return (
    <div className="container">
      <div className="header">
        <h1><i className="fas fa-search"></i> Job Analysis Dashboard</h1>
        <p>Analyze job postings against your experience and skills</p>
      </div>

      <div className="content">
        <JobForm onSubmit={handleFormSubmit} />
        
        {currentJob && (
          <div className="status-section">
            <h2><i className="fas fa-chart-line"></i> Job Status</h2>
            <JobStatus job={currentJob} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 