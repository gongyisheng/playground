import React from 'react';

const JobResults = ({ results }) => {
  return (
    <div className="results-section">
      <h3>Results ({results.length} jobs)</h3>
      {results.map((job, index) => (
        <JobCard key={index} job={job} />
      ))}
    </div>
  );
};

const JobCard = ({ job }) => {
  return (
    <div className={`job-card ${job.is_match ? 'match' : 'no-match'}`}>
      <div className="job-title">{job.position_name}</div>
      <div className="job-company">{job.company_name}</div>
      <div className="job-company">Experience: {job.min_years_experience}</div>
      <div className="job-company">
        <a 
          href={job.job_link} 
          target="_blank" 
          rel="noopener noreferrer"
          style={{ color: '#667eea' }}
        >
          View Job
        </a>
      </div>
      
      {job.matched_skills && job.matched_skills.length > 0 && (
        <div className="skills-section">
          <strong>Matched Skills:</strong>
          {job.matched_skills.map((skill, index) => (
            <span key={index} className="skill-tag matched">
              {skill}
            </span>
          ))}
        </div>
      )}
      
      {job.miss_skills && job.miss_skills.length > 0 && (
        <div className="skills-section">
          <strong>Missing Skills:</strong>
          {job.miss_skills.map((skill, index) => (
            <span key={index} className="skill-tag missed">
              {skill}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export default JobResults; 