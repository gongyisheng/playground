import tornado.ioloop
import tornado.web
import json
from crawler import run_crawler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CrawlerHandler(tornado.web.RequestHandler):
    def get(self):
        """
        GET endpoint to trigger the LinkedIn job crawler
        Returns:
            JSON response with job details
        """
        try:
            logging.info("Starting job crawler")
            job_details = run_crawler()
            
            # Prepare response
            response = {
                "status": "success",
                "message": "Successfully crawled jobs",
                "data": job_details
            }
            
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(response, indent=2))
            logging.info(f"Crawled {len(job_details)} jobs successfully")
            
        except Exception as e:
            logging.error(f"Error in crawler: {str(e)}", exc_info=True)
            self.set_status(500)
            self.write(json.dumps({
                "status": "error",
                "message": f"Error crawling jobs: {str(e)}"
            }))

def make_app():
    """
    Create and configure the Tornado application
    """
    return tornado.web.Application([
        (r"/api/crawl", CrawlerHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    port = 8888
    logging.info(f"Starting server on port {port}")
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()
