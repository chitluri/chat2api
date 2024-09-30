import random
import logging

from curl_cffi.requests import AsyncSession

# Constants
DEFAULT_TIMEOUT = 15
IMPERSONATION_OPTIONS = ['safari15_3']
DEFAULT_IMPERSONATE = random.choice(IMPERSONATION_OPTIONS)

# Logger setup
logger = logging.getLogger(__name__)

class Client:
    def __init__(self, proxy=None, timeout=DEFAULT_TIMEOUT, verify=True):
        """
        Initializes the Client object, which sets up an HTTP client session
        with optional proxy, timeout, and SSL verification settings.
        """
        self.proxies = {"http": proxy, "https": proxy}
        self.timeout = timeout
        self.verify = verify
        self.impersonate = DEFAULT_IMPERSONATE
        
        # Create a single session instance for all requests
        self.session = AsyncSession(proxies=self.proxies, timeout=self.timeout, verify=self.verify)

    async def _make_request(self, method, *args, headers=None, cookies=None, **kwargs):
        """
        A generic method to make HTTP requests, reducing code duplication.
        
        Parameters:
        - method (str): HTTP method such as 'get', 'post', 'put', etc.
        - *args, **kwargs: Additional arguments passed to the session request.
        
        Returns:
        - Response object from the HTTP request.
        """
        try:
            headers = headers or self.session.headers
            cookies = cookies or self.session.cookies
            
            request_method = getattr(self.session, method.lower())
            response = await request_method(*args, headers=headers, cookies=cookies, impersonate=self.impersonate, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Error making {method.upper()} request: {e}")
            raise

    async def post(self, *args, **kwargs):
        """
        Makes an asynchronous HTTP POST request using the session.
        """
        return await self._make_request('post', *args, **kwargs)

    async def get(self, *args, **kwargs):
        """
        Makes an asynchronous HTTP GET request using the session.
        """
        return await self._make_request('get', *args, **kwargs)

    async def request(self, *args, **kwargs):
        """
        Makes an asynchronous HTTP request using the session with custom method.
        """
        return await self._make_request('request', *args, **kwargs)

    async def put(self, *args, **kwargs):
        """
        Makes an asynchronous HTTP PUT request using the session.
        """
        return await self._make_request('put', *args, **kwargs)

    async def post_stream(self, *args, headers=None, cookies=None, **kwargs):
        """
        Makes an asynchronous HTTP POST request specifically for streaming.
        It allows headers and cookies to be passed separately.
        """
        # Handle headers and cookies separately for streaming requests
        headers = headers or self.session.headers
        cookies = cookies or self.session.cookies
        return await self._make_request('post', *args, headers=headers, cookies=cookies, **kwargs)

    async def close(self):
        """
        Closes the session to clean up resources. Ensures that the session is closed
        properly and logs any exceptions that occur during the process.
        """
        if self.session:
            try:
                await self.session.close()
                logger.info("Session closed successfully")
            except Exception as e:
                logger.error(f"Error closing session: {e}")
                raise

# Example usage (should be placed in your main program or test suite, not in the class definition)
# client = Client(proxy="http://example.com")
# response = await client.post("https://httpbin.org/post", json={"data": "example"})
# await client.close()
