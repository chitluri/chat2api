import asyncio
from typing import Optional

from fastapi import HTTPException

from chatgpt.refreshToken import rt2ac
from utils.Logger import logger
from utils.config import authorization_list
import chatgpt.globals as globals

# Constants for token validation
TOKEN_LENGTH = 45
JWT_PREFIX = "eyJhbGciOi"
FK_PREFIX = "fk-"
TOKEN_REFRESH_DELAY = 2  # seconds


class TokenService:
    """
    Service responsible for managing and verifying authorization tokens.
    Encapsulates token retrieval, verification, and refreshing mechanisms.
    """

    def __init__(self):
        """
        Initializes the TokenService with necessary configurations and state.
        """
        self.authorization_list = authorization_list
        self.token_list = globals.token_list
        self.error_token_list = globals.error_token_list
        self.count = globals.count

    def get_required_token(self, requested_token: str) -> Optional[str]:
        """
        Retrieves a valid token based on the requested token and authorization list.

        Args:
            requested_token (str): The token provided in the request.

        Returns:
            Optional[str]: A valid token if available; otherwise, None or the requested token.
        """
        if requested_token in self.authorization_list:
            available_tokens = len(self.token_list) - len(self.error_token_list)
            if available_tokens > 0:
                self.count = (self.count + 1) % len(self.token_list)
                # Iterate to find the next valid token not in error_token_list
                original_count = self.count
                while self.token_list[self.count] in self.error_token_list:
                    self.count = (self.count + 1) % len(self.token_list)
                    if self.count == original_count:
                        # All tokens are in error_token_list
                        logger.warning("All tokens are in the error list.")
                        return None
                return self.token_list[self.count]
            else:
                logger.warning("No available tokens to provide.")
                return None
        else:
            return requested_token

    async def verify_token(self, token: Optional[str]) -> Optional[str]:
        """
        Verifies the provided token and retrieves an access token if necessary.

        Args:
            token (Optional[str]): The token to verify.

        Returns:
            Optional[str]: A valid access token or raises an HTTPException for unauthorized access.
        """
        if not token:
            if self.authorization_list:
                logger.error("Unauthorized access attempt with empty token.")
                raise HTTPException(status_code=401, detail="Unauthorized: Empty token.")
            else:
                return None

        if token.startswith(JWT_PREFIX) or token.startswith(FK_PREFIX):
            # Token is already a valid access token
            return token
        elif len(token) == TOKEN_LENGTH:
            try:
                access_token = await rt2ac(token, force_refresh=False)
                return access_token
            except HTTPException as e:
                logger.error(f"Token verification failed: {e.detail}")
                raise HTTPException(status_code=e.status_code, detail=e.detail) from e
            except Exception as e:
                logger.exception("An unexpected error occurred during token verification.")
                raise HTTPException(status_code=500, detail="Internal Server Error") from e
        else:
            logger.warning(f"Invalid token format received: {token}")
            return token

    async def refresh_all_tokens(self, force_refresh: bool = False) -> None:
        """
        Refreshes all tokens that require refreshing asynchronously.

        Args:
            force_refresh (bool): Indicates whether to force refresh tokens regardless of their state.
        """
        async def refresh_token(token: str) -> None:
            """
            Refreshes a single token.

            Args:
                token (str): The token to refresh.
            """
            try:
                await asyncio.sleep(TOKEN_REFRESH_DELAY)
                await rt2ac(token, force_refresh=force_refresh)
                logger.info(f"Token refreshed successfully: {token}")
            except HTTPException as e:
                logger.error(f"Failed to refresh token {token}: {e.detail}")
            except Exception as e:
                logger.exception(f"Unexpected error while refreshing token {token}.")

        # Gather all refresh tasks concurrently
        refresh_tasks = [
            refresh_token(token)
            for token in self.token_list
            if len(token) == TOKEN_LENGTH
        ]
        if refresh_tasks:
            await asyncio.gather(*refresh_tasks)
            logger.info("All token refresh operations completed.")
        else:
            logger.info("No tokens require refreshing.")


# Instantiate the TokenService for use in other modules
token_service = TokenService()

# Example usage within FastAPI routes (not included in the original code)
# from fastapi import Depends

# @app.get("/protected-route")
# async def protected_route(token: Optional[str] = Depends(get_token_dependency)):
#     access_token = await token_service.verify_token(token)
#     if not access_token:
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     # Proceed with the route logic using access_token
