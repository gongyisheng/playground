import React from 'react';
import JobResults from './JobResults';

const JobStatus = ({ job }) => {
  const progressPercent = job.total > 0 ? (job.progress / job.total) * 100 : 0;

  return (
    <div className={`job-status ${job.status}`}>
      <div className="job-header">
        <span className="job-id">{job.job_id}</span>
        <span className={`job-status-badge status-${job.status}`}>
          {job.status}
        </span>
      </div>
      
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${progressPercent}%` }}
        ></div>
      </div>
      
      <div className="job-message">
        {job.message} ({job.progress}/{job.total})
      </div>
      
      {job.error && (
        <div className="job-message" style={{ color: '#dc3545' }}>
          Error: {job.error}
        </div>
      )}
      
      {job.results && job.results.length > 0 && (
        <JobResults results={job.results} />
      )}
    </div>
  );
};

export default JobStatus; 