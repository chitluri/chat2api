import uvicorn
import logging
import os

def get_logging_config():
    """
    Configures the logging format for both default and access logs.

    Returns:
        dict: The updated logging configuration for Uvicorn.
    """
    log_config = uvicorn.config.LOGGING_CONFIG.copy()  # Copy to avoid modifying global config
    
    default_format = "%(asctime)s | %(levelname)s | %(message)s"
    access_format = r'%(asctime)s | %(levelname)s | %(client_addr)s: %(request_line)s %(status_code)s'
    
    # Update the formatters in the logging configuration
    log_config["formatters"]["default"]["fmt"] = default_format
    log_config["formatters"]["access"]["fmt"] = access_format
    
    return log_config

def run_server():
    """
    Runs the Uvicorn server using custom logging configurations.
    Configuration for host and port is retrieved from environment variables.
    """
    # Fetch host and port from environment variables, with defaults
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", 5005))
    
    # Get the custom logging configuration
    logging_config = get_logging_config()
    
    # Log the server start attempt
    logging.info(f"Starting server at {host}:{port}")
    
    try:
        # Run the Uvicorn server with custom logging
        uvicorn.run("chat2api:app", host=host, port=port, log_config=logging_config)
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    # Entry point for the script
    run_server()