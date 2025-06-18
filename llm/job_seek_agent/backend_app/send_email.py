import boto3
import json
import logging
from typing import List, Dict, Optional, Union
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SESEmailService:
    def __init__(self, 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-east-1',
                 sender_email: Optional[str] = None):
        """
        Initialize AWS SES email service
        
        Args:
            aws_access_key_id: AWS access key ID (optional, uses environment variables if not provided)
            aws_secret_access_key: AWS secret access key (optional, uses environment variables if not provided)
            region_name: AWS region name (default: us-east-1)
            sender_email: Verified sender email address
        """
        self.aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region_name = region_name
        self.sender_email = sender_email or os.getenv('SES_SENDER_EMAIL')
        
        if not self.aws_access_key_id or not self.aws_secret_access_key:
            raise ValueError("AWS credentials must be provided via parameters or environment variables")
        
        if not self.sender_email:
            raise ValueError("Sender email must be provided via parameter or SES_SENDER_EMAIL environment variable")
        
        # Initialize SES client
        self.ses_client = boto3.client(
            'ses',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
        )
        
        logging.info(f"SES Email Service initialized for region: {self.region_name}")

    def verify_email_identity(self, email_address: str) -> bool:
        """
        Verify an email address with AWS SES
        
        Args:
            email_address: Email address to verify
            
        Returns:
            bool: True if verification request was sent successfully
        """
        try:
            response = self.ses_client.verify_email_identity(
                EmailAddress=email_address
            )
            logging.info(f"Verification email sent to: {email_address}")
            return True
        except Exception as e:
            logging.error(f"Error verifying email {email_address}: {str(e)}")
            return False

    def check_verification_status(self, email_address: str) -> str:
        """
        Check the verification status of an email address
        
        Args:
            email_address: Email address to check
            
        Returns:
            str: Verification status ('Pending', 'Success', 'Failed', 'TemporaryFailure')
        """
        try:
            response = self.ses_client.get_identity_verification_attributes(
                Identities=[email_address]
            )
            
            if email_address in response['VerificationAttributes']:
                return response['VerificationAttributes'][email_address]['VerificationStatus']
            else:
                return 'NotVerified'
        except Exception as e:
            logging.error(f"Error checking verification status for {email_address}: {str(e)}")
            return 'Error'

    def get_sending_quota(self) -> Dict[str, int]:
        """
        Get the current sending quota for the AWS account
        
        Returns:
            Dict containing max sending rate and max 24 hour send
        """
        try:
            response = self.ses_client.get_send_quota()
            return {
                'max_sending_rate': response['MaxSendRate'],
                'max_24_hour_send': response['Max24HourSend'],
                'sent_last_24_hours': response['SentLast24Hours']
            }
        except Exception as e:
            logging.error(f"Error getting sending quota: {str(e)}")
            return {}

    def send_job_analysis_results(self, 
                                 recipient_email: str,
                                 job_results: List[Dict],
                                 search_params: Dict,
                                 total_jobs: int,
                                 matched_jobs: int) -> bool:
        """
        Send job analysis results via email
        
        Args:
            recipient_email: Email address of the recipient
            job_results: List of job analysis results
            search_params: Search parameters used for the analysis
            total_jobs: Total number of jobs analyzed
            matched_jobs: Number of jobs that matched the candidate
            
        Returns:
            bool: True if email was sent successfully
        """
        try:
            # Create email content
            subject = f"Job Analysis Results - {matched_jobs} matches found"
            
            # Generate HTML content
            html_content = self._generate_job_results_html(
                job_results, search_params, total_jobs, matched_jobs
            )
            
            # Generate plain text content
            text_content = self._generate_job_results_text(
                job_results, search_params, total_jobs, matched_jobs
            )
            
            # Send email
            return self._send_email(
                recipient_email=recipient_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logging.error(f"Error sending job analysis results: {str(e)}")
            return False

    def send_job_notification(self, 
                            recipient_email: str,
                            job_id: str,
                            status: str,
                            message: str,
                            progress: int = 0,
                            total: int = 0) -> bool:
        """
        Send job status notification
        
        Args:
            recipient_email: Email address of the recipient
            job_id: Job ID
            status: Current job status
            message: Status message
            progress: Current progress
            total: Total jobs to process
            
        Returns:
            bool: True if email was sent successfully
        """
        try:
            subject = f"Job Analysis Update - {job_id}"
            
            html_content = self._generate_notification_html(
                job_id, status, message, progress, total
            )
            
            text_content = self._generate_notification_text(
                job_id, status, message, progress, total
            )
            
            return self._send_email(
                recipient_email=recipient_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
        except Exception as e:
            logging.error(f"Error sending job notification: {str(e)}")
            return False

    def send_custom_email(self, 
                         recipient_email: str,
                         subject: str,
                         html_content: str,
                         text_content: Optional[str] = None,
                         attachments: Optional[List[Dict]] = None) -> bool:
        """
        Send a custom email with optional attachments
        
        Args:
            recipient_email: Email address of the recipient
            subject: Email subject
            html_content: HTML content of the email
            text_content: Plain text content (optional)
            attachments: List of attachment dictionaries with 'filename' and 'content' keys
            
        Returns:
            bool: True if email was sent successfully
        """
        try:
            return self._send_email(
                recipient_email=recipient_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                attachments=attachments
            )
        except Exception as e:
            logging.error(f"Error sending custom email: {str(e)}")
            return False

    def _send_email(self, 
                   recipient_email: str,
                   subject: str,
                   html_content: str,
                   text_content: Optional[str] = None,
                   attachments: Optional[List[Dict]] = None) -> bool:
        """
        Internal method to send email via AWS SES
        
        Args:
            recipient_email: Email address of the recipient
            subject: Email subject
            html_content: HTML content
            text_content: Plain text content
            attachments: List of attachments
            
        Returns:
            bool: True if email was sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            
            # Add text content
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)
            
            # Send email via SES
            response = self.ses_client.send_raw_email(
                Source=self.sender_email,
                Destinations=[recipient_email],
                RawMessage={'Data': msg.as_string()}
            )
            
            logging.info(f"Email sent successfully to {recipient_email}. Message ID: {response['MessageId']}")
            return True
            
        except Exception as e:
            logging.error(f"Error sending email to {recipient_email}: {str(e)}")
            return False

    def _generate_job_results_html(self, 
                                  job_results: List[Dict],
                                  search_params: Dict,
                                  total_jobs: int,
                                  matched_jobs: int) -> str:
        """Generate HTML content for job results email"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .job-card {{ background: white; border: 1px solid #e1e5e9; border-radius: 8px; padding: 15px; margin-bottom: 15px; }}
                .job-card.match {{ border-left: 4px solid #28a745; }}
                .job-card.no-match {{ border-left: 4px solid #dc3545; }}
                .job-title {{ font-weight: bold; color: #333; margin-bottom: 5px; }}
                .job-company {{ color: #666; font-size: 14px; margin-bottom: 5px; }}
                .skill-tag {{ background: #667eea; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; margin-right: 5px; }}
                .skill-tag.matched {{ background: #28a745; }}
                .skill-tag.missed {{ background: #dc3545; }}
                .footer {{ background: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Job Analysis Results</h1>
                <p>Your personalized job analysis is complete!</p>
            </div>
            
            <div class="content">
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Total Jobs Analyzed:</strong> {total_jobs}</p>
                    <p><strong>Matching Jobs Found:</strong> {matched_jobs}</p>
                    <p><strong>Search Keywords:</strong> {search_params.get('keywords', 'N/A')}</p>
                    <p><strong>Location:</strong> {search_params.get('location', 'N/A')}</p>
                    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>Job Results</h2>
        """
        
        for job in job_results:
            match_class = 'match' if job.get('is_match') else 'no-match'
            html += f"""
                <div class="job-card {match_class}">
                    <div class="job-title">{job.get('position_name', 'N/A')}</div>
                    <div class="job-company">{job.get('company_name', 'N/A')}</div>
                    <div class="job-company">Experience: {job.get('min_years_experience', 'N/A')}</div>
                    <div class="job-company">
                        <a href="{job.get('job_link', '#')}" target="_blank" style="color: #667eea;">View Job</a>
                    </div>
            """
            
            if job.get('matched_skills'):
                html += '<div><strong>Matched Skills:</strong> '
                for skill in job['matched_skills']:
                    html += f'<span class="skill-tag matched">{skill}</span>'
                html += '</div>'
            
            if job.get('miss_skills'):
                html += '<div><strong>Missing Skills:</strong> '
                for skill in job['miss_skills']:
                    html += f'<span class="skill-tag missed">{skill}</span>'
                html += '</div>'
            
            html += '</div>'
        
        html += """
            </div>
            
            <div class="footer">
                <p>This email was generated by the Job Analysis System</p>
                <p>Powered by AWS SES and OpenAI GPT-4</p>
            </div>
        </body>
        </html>
        """
        
        return html

    def _generate_job_results_text(self, 
                                  job_results: List[Dict],
                                  search_params: Dict,
                                  total_jobs: int,
                                  matched_jobs: int) -> str:
        """Generate plain text content for job results email"""
        text = f"""
Job Analysis Results

Summary:
- Total Jobs Analyzed: {total_jobs}
- Matching Jobs Found: {matched_jobs}
- Search Keywords: {search_params.get('keywords', 'N/A')}
- Location: {search_params.get('location', 'N/A')}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Job Results:
"""
        
        for i, job in enumerate(job_results, 1):
            match_status = "MATCH" if job.get('is_match') else "NO MATCH"
            text += f"""
{i}. {job.get('position_name', 'N/A')} - {match_status}
   Company: {job.get('company_name', 'N/A')}
   Experience: {job.get('min_years_experience', 'N/A')}
   Job Link: {job.get('job_link', 'N/A')}
"""
            
            if job.get('matched_skills'):
                text += f"   Matched Skills: {', '.join(job['matched_skills'])}\n"
            
            if job.get('miss_skills'):
                text += f"   Missing Skills: {', '.join(job['miss_skills'])}\n"
        
        text += """
---
This email was generated by the Job Analysis System
Powered by AWS SES and OpenAI GPT-4
"""
        
        return text

    def _generate_notification_html(self, 
                                   job_id: str,
                                   status: str,
                                   message: str,
                                   progress: int,
                                   total: int) -> str:
        """Generate HTML content for job notification email"""
        progress_percent = (progress / total * 100) if total > 0 else 0
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .status-card {{ background: white; border: 1px solid #e1e5e9; border-radius: 8px; padding: 15px; margin-bottom: 15px; }}
                .progress-bar {{ width: 100%; height: 20px; background: #e1e5e9; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
                .progress-fill {{ height: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); transition: width 0.3s ease; }}
                .footer {{ background: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Job Analysis Update</h1>
                <p>Your job analysis is in progress</p>
            </div>
            
            <div class="content">
                <div class="status-card">
                    <h2>Job ID: {job_id}</h2>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Message:</strong> {message}</p>
                    <p><strong>Progress:</strong> {progress} / {total} ({progress_percent:.1f}%)</p>
                    
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress_percent}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>This email was generated by the Job Analysis System</p>
                <p>Powered by AWS SES</p>
            </div>
        </body>
        </html>
        """
        
        return html

    def _generate_notification_text(self, 
                                   job_id: str,
                                   status: str,
                                   message: str,
                                   progress: int,
                                   total: int) -> str:
        """Generate plain text content for job notification email"""
        progress_percent = (progress / total * 100) if total > 0 else 0
        
        text = f"""
Job Analysis Update

Job ID: {job_id}
Status: {status}
Message: {message}
Progress: {progress} / {total} ({progress_percent:.1f}%)

---
This email was generated by the Job Analysis System
Powered by AWS SES
"""
        
        return text

# Example usage and utility functions
def create_email_service_from_env() -> SESEmailService:
    """
    Create SES email service using environment variables
    
    Required environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - SES_SENDER_EMAIL
    
    Optional environment variables:
    - AWS_DEFAULT_REGION (default: us-east-1)
    
    Returns:
        SESEmailService instance
    """
    region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    return SESEmailService(region_name=region)

def send_job_completion_notification(recipient_email: str, 
                                    job_results: List[Dict],
                                    search_params: Dict) -> bool:
    """
    Utility function to send job completion notification
    
    Args:
        recipient_email: Email address to send results to
        job_results: List of job analysis results
        search_params: Search parameters used
        
    Returns:
        bool: True if email was sent successfully
    """
    try:
        email_service = create_email_service_from_env()
        
        total_jobs = len(job_results)
        matched_jobs = sum(1 for job in job_results if job.get('is_match', False))
        
        return email_service.send_job_analysis_results(
            recipient_email=recipient_email,
            job_results=job_results,
            search_params=search_params,
            total_jobs=total_jobs,
            matched_jobs=matched_jobs
        )
    except Exception as e:
        logging.error(f"Error sending job completion notification: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    try:
        # Create email service
        email_service = create_email_service_from_env()
        
        # Check sending quota
        quota = email_service.get_sending_quota()
        print(f"Sending quota: {quota}")
        
        # Example: Send a test email
        test_email = "your-email@example.com"  # Replace with actual email
        
        # Verify email first (only needed once)
        if email_service.verify_email_identity(test_email):
            print(f"Verification email sent to {test_email}")
            print("Please check your email and click the verification link")
        
        # Send a test notification
        success = email_service.send_job_notification(
            recipient_email=test_email,
            job_id="test_job_123",
            status="completed",
            message="Test job analysis completed successfully",
            progress=10,
            total=10
        )
        
        if success:
            print("Test email sent successfully!")
        else:
            print("Failed to send test email")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have set the required environment variables:")
        print("- AWS_ACCESS_KEY_ID")
        print("- AWS_SECRET_ACCESS_KEY")
        print("- SES_SENDER_EMAIL") 