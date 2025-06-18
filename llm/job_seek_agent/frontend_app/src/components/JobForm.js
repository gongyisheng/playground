import React, { useState } from 'react';

const JobForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    keywords: 'Software Engineer',
    location: 'San Francisco Bay Area',
    limit: 10,
    f_E: '',
    f_JT: '',
    f_WT: '',
    f_TPR: '',
    f_AL: false,
    f_JIYN: false
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      await onSubmit(formData);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="form-section">
      <h2><i className="fas fa-cog"></i> Job Search Configuration</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label htmlFor="keywords">Keywords</label>
            <input
              type="text"
              id="keywords"
              name="keywords"
              value={formData.keywords}
              onChange={handleInputChange}
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="location">Location</label>
            <input
              type="text"
              id="location"
              name="location"
              value={formData.location}
              onChange={handleInputChange}
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="limit">Number of Jobs</label>
            <input
              type="number"
              id="limit"
              name="limit"
              value={formData.limit}
              onChange={handleInputChange}
              min="1"
              max="100"
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="f_E">Experience Level</label>
            <select
              id="f_E"
              name="f_E"
              value={formData.f_E}
              onChange={handleInputChange}
            >
              <option value="">Any Level</option>
              <option value="1">Intern</option>
              <option value="2">Assistant</option>
              <option value="3">Junior</option>
              <option value="4">Mid-Senior</option>
              <option value="5">Director</option>
              <option value="6">Executive</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="f_JT">Job Type</label>
            <select
              id="f_JT"
              name="f_JT"
              value={formData.f_JT}
              onChange={handleInputChange}
            >
              <option value="">Any Type</option>
              <option value="F">Full-time</option>
              <option value="P">Part-time</option>
              <option value="C">Contract</option>
              <option value="T">Temporary</option>
              <option value="V">Volunteer</option>
              <option value="I">Internship</option>
              <option value="O">Other</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="f_WT">Work Type</label>
            <select
              id="f_WT"
              name="f_WT"
              value={formData.f_WT}
              onChange={handleInputChange}
            >
              <option value="">Any Work Type</option>
              <option value="1">On-site</option>
              <option value="2">Remote</option>
              <option value="3">Hybrid</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="f_TPR">Time Posted</label>
            <select
              id="f_TPR"
              name="f_TPR"
              value={formData.f_TPR}
              onChange={handleInputChange}
            >
              <option value="">Any Time</option>
              <option value="r86400">Past 24 Hours</option>
              <option value="r604800">Past Week</option>
              <option value="r2592000">Past Month</option>
            </select>
          </div>
        </div>

        <div className="checkbox-group">
          <input
            type="checkbox"
            id="f_AL"
            name="f_AL"
            checked={formData.f_AL}
            onChange={handleInputChange}
          />
          <label htmlFor="f_AL">Easy Apply Only</label>
        </div>

        <div className="checkbox-group">
          <input
            type="checkbox"
            id="f_JIYN"
            name="f_JIYN"
            checked={formData.f_JIYN}
            onChange={handleInputChange}
          />
          <label htmlFor="f_JIYN">Less than 10 applicants</label>
        </div>

        <button type="submit" className="btn" disabled={isSubmitting}>
          {isSubmitting ? (
            <>
              <div className="loading"></div>
              Starting...
            </>
          ) : (
            <>
              <i className="fas fa-play"></i>
              Start Analysis
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default JobForm; 