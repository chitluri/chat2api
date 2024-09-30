import asyncio
import json
import random
import uuid
from typing import Optional, List, Dict, Any

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.files import get_image_size, get_file_extension, determine_file_use_case
from api.models import model_proxy
# from chatgpt.authorization import get_req_token, verify_token
from chatgpt.authorization import token_service
from chatgpt.chatFormat import (
    api_messages_to_chat,
    stream_response,
    format_not_stream_response,
    head_process_response,
)
from chatgpt.chatLimit import check_is_limit, handle_request_limit
from chatgpt.proofofWork import get_config, get_dpl, get_answer_token, get_requirements_token
from chatgpt.turnstile import process_turnstile
from utils.Client import Client
from utils.Logger import logger
from utils.config import (
    proxy_url_list,
    chatgpt_base_url_list,
    ark0se_token_url_list,
    history_disabled,
    pow_difficulty,
    conversation_only,
    enable_limit,
    upload_by_url,
    check_model,
    auth_key,
    user_agents_list,
)


class ChatService:
    """
    A service class to handle chat interactions with the OpenAI API.
    """

    def __init__(self, origin_token: Optional[str] = None):
        """
        Initializes the ChatService instance.

        :param origin_token: The original token provided for authentication.
        """
        # Select a random user agent or use the default if none are provided
        default_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        )
        self.user_agent = random.choice(user_agents_list) if user_agents_list else default_user_agent

        # Initialize tokens and client
        self.req_token = token_service.get_required_token(origin_token)
        self.chat_token = "gAAAAAB"
        self.access_token: Optional[str] = None
        self.account_id: Optional[str] = None
        self.s: Optional[Client] = None
        self.ws = None

        # Initialize other attributes
        self.data: Dict[str, Any] = {}
        self.origin_model: str = ""
        self.resp_model: str = ""
        self.req_model: str = ""
        self.prompt_tokens: int = 0
        self.max_tokens: int = 2147483647
        self.api_messages: List[Dict[str, Any]] = []
        self.persona: Optional[str] = None
        self.ark0se_token: Optional[str] = None
        self.proof_token: Optional[str] = None
        self.turnstile_token: Optional[str] = None
        self.chat_headers: Optional[Dict[str, str]] = None
        self.chat_request: Optional[Dict[str, Any]] = None
        self.proxy_url: Optional[str] = None
        self.host_url: str = ""
        self.ark0se_token_url: Optional[str] = None
        self.oai_device_id: str = str(uuid.uuid4())
        self.base_headers: Dict[str, str] = {}
        self.base_url: str = ""
        self.conversation_id: Optional[str] = None
        self.parent_message_id: Optional[str] = None
        self.history_disabled: bool = history_disabled

        # Initialize other configurations
        self.initialize_configurations()

    def initialize_configurations(self):
        """
        Initializes configurations such as proxy URLs, host URLs, and headers.
        """
        # Set proxy URL, host URL, and Arkose token URL
        self.proxy_url = random.choice(proxy_url_list) if proxy_url_list else None
        self.host_url = random.choice(chatgpt_base_url_list) if chatgpt_base_url_list else "https://chatgpt.com"
        self.ark0se_token_url = random.choice(ark0se_token_url_list) if ark0se_token_url_list else None

        # Initialize the HTTP client with proxy
        self.s = Client(proxy=self.proxy_url)

        # Set base headers
        self.base_headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Accept-Language': 'en-US,en;q=0.9',
            'Content-Type': 'application/json',
            'Oai-Device-Id': self.oai_device_id,
            'Oai-Language': 'en-US',
            'Origin': self.host_url,
            'Priority': 'u=1, i',
            'Referer': f'{self.host_url}/',
            'Sec-Ch-Ua': '"Chromium";v="124", "Microsoft Edge";v="124", "Not-A.Brand";v="99"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': self.user_agent
        }

        # Set base URL and authorization if access_token is available
        if self.access_token:
            self.base_url = f"{self.host_url}/backend-api"
            self.base_headers['Authorization'] = f'Bearer {self.access_token}'
            if self.account_id:
                self.base_headers['Chatgpt-Account-Id'] = self.account_id
        else:
            self.base_url = f"{self.host_url}/backend-anon"

        # Add auth_key if provided
        if auth_key:
            self.base_headers['authkey'] = auth_key

        # Set cookies
        domain = self.host_url.split("://")[1]
        self.s.session.cookies.set(
            "__Secure-next-auth.callback-url", "https%3A%2F%2Fchatgpt.com;",
            domain=domain, secure=True
        )

    async def set_dynamic_data(self, data: Dict[str, Any]):
        """
        Sets dynamic data for the ChatService instance based on the provided data.

        :param data: A dictionary containing request data.
        """
        self.data = data
        # Handle authentication tokens
        await self.handle_authentication()

        # Set model configurations
        await self.set_model()

        # Handle request limits if enabled
        if enable_limit and self.req_token:
            limit_response = await handle_request_limit(self.req_token, self.req_model)
            if limit_response:
                raise HTTPException(status_code=429, detail=limit_response)

        # Set conversation parameters
        self.set_conversation_parameters()

        # Set API messages and token counts
        self.set_messages_and_tokens()

        # Re-initialize configurations based on updated access_token
        self.initialize_configurations()

        # Perform device-related setup
        await get_dpl(self)

    async def handle_authentication(self):
        """
        Handles authentication by verifying the request token and setting access tokens.
        """
        if self.req_token:
            logger.info(f"Request token: {self.req_token}")
            req_tokens = self.req_token.split(",")
            if len(req_tokens) == 1:
                self.access_token = await token_service.verify_token(self.req_token)
                self.account_id = None
            else:
                self.access_token = await token_service.verify_token(req_tokens[0])
                self.account_id = req_tokens[1]
        else:
            logger.info("Request token is empty, using no-auth 3.5")
            self.access_token = None
            self.account_id = None

    def set_conversation_parameters(self):
        """
        Sets conversation-related parameters from the request data.
        """
        self.account_id = self.data.get('Chatgpt-Account-Id', self.account_id)
        self.parent_message_id = self.data.get('parent_message_id', str(uuid.uuid4()))
        self.conversation_id = self.data.get('conversation_id')
        self.history_disabled = self.data.get('history_disabled', history_disabled)

    def set_messages_and_tokens(self):
        """
        Sets the messages and token counts from the request data.
        """
        self.api_messages = self.data.get("messages", [])
        self.prompt_tokens = 0
        max_tokens = self.data.get("max_tokens", 2147483647)
        self.max_tokens = max_tokens if isinstance(max_tokens, int) else 2147483647

    async def set_model(self):
        """
        Sets the model configurations based on the request data.
        """
        self.origin_model = self.data.get("model", "gpt-3.5-turbo-0125")
        self.resp_model = model_proxy.get(self.origin_model, self.origin_model)
        self.req_model = self.determine_req_model(self.origin_model)

    def determine_req_model(self, model_name: str) -> str:
        """
        Determines the required model based on the origin model name.

        :param model_name: The name of the original model.
        :return: The required model name.
        """
        if "o1-preview" in model_name:
            return "o1-preview"
        elif "o1-mini" in model_name:
            return "o1-mini"
        elif "o1" in model_name:
            return "o1"
        elif "gpt-4o-mini" in model_name:
            return "gpt-4o-mini"
        elif "gpt-4o" in model_name:
            return "gpt-4o"
        elif "gpt-4-mobile" in model_name:
            return "gpt-4-mobile"
        elif "gpt-4-gizmo" in model_name:
            return "gpt-4o"
        elif "gpt-4" in model_name:
            return "gpt-4"
        elif "gpt-3.5" in model_name:
            return "text-davinci-002-render-sha"
        elif "auto" in model_name:
            return "auto"
        else:
            return "auto"

    async def get_chat_requirements(self):
        """
        Retrieves chat requirements such as tokens and performs necessary checks.

        :return: The chat token if successful, else raises an HTTPException.
        """
        if conversation_only:
            return None

        url = f'{self.base_url}/sentinel/chat-requirements'
        headers = self.base_headers.copy()

        try:
            # Get the configuration and requirements token
            config = get_config(self.user_agent)
            p = get_requirements_token(config)
            data = {'p': p}

            # Send POST request to get chat requirements
            response = await self.s.post(url, headers=headers, json=data, timeout=5)
            if response.status_code == 200:
                resp_json = response.json()
                await self.process_chat_requirements_response(resp_json, p, headers)
                return self.chat_token
            else:
                await self.handle_unsuccessful_chat_requirements(response)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def process_chat_requirements_response(self, resp_json: Dict[str, Any], p: str, headers: Dict[str, str]):
        """
        Processes the response from the chat requirements request.

        :param resp_json: The JSON response from the server.
        :param p: The requirements token.
        :param headers: The request headers.
        """
        if check_model:
            await self.verify_model_availability(headers)
        else:
            self.persona = resp_json.get("persona")
            if self.persona != "chatgpt-paid" and self.req_model == "gpt-4":
                logger.error(f"Model {self.req_model} not supported for {self.persona}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": f"The model {self.origin_model} does not exist or you do not have access to it.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "model_not_found"
                    }
                )

        # Process turnstile if required
        await self.process_turnstile(resp_json.get('turnstile', {}), p)

        # Process Arkose if required
        await self.process_arkose(resp_json.get('arkose', {}))

        # Process proof of work if required
        await self.process_proof_of_work(resp_json.get('proofofwork', {}))

        # Set the chat token
        self.chat_token = resp_json.get('token')
        if not self.chat_token:
            raise HTTPException(status_code=403, detail="Failed to get chat token")

    async def verify_model_availability(self, headers: Dict[str, str]):
        """
        Verifies if the requested model is available.

        :param headers: The request headers.
        """
        models_url = f'{self.base_url}/models'
        response = await self.s.get(models_url, headers=headers, timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if not any(self.req_model in model.get("slug", "") for model in models):
                logger.error(f"Model {self.req_model} not supported.")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": f"The model {self.origin_model} does not exist or you do not have access to it.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "model_not_found"
                    }
                )
        else:
            raise HTTPException(status_code=404, detail="Failed to get models")

    async def process_turnstile(self, turnstile: Dict[str, Any], p: str):
        """
        Processes the turnstile challenge if required.

        :param turnstile: The turnstile information from the response.
        :param p: The requirements token.
        """
        if turnstile.get('required'):
            turnstile_dx = turnstile.get("dx")
            try:
                self.turnstile_token = process_turnstile(turnstile_dx, p)
            except Exception as e:
                logger.info(f"Turnstile ignored: {e}")
                # Optionally, handle the exception or raise an error
                # For now, we ignore the failure
                pass

    async def process_arkose(self, arkose: Dict[str, Any]):
        """
        Processes the Arkose challenge if required.

        :param arkose: The Arkose information from the response.
        """
        if arkose.get('required'):
            if self.persona == "chatgpt-freeaccount":
                arkose_method = "chat35"
            else:
                arkose_method = "chat4"

            if not self.ark0se_token_url:
                raise HTTPException(status_code=403, detail="Arkose service required")

            arkose_dx = arkose.get("dx")
            arkose_client = Client()
            try:
                response = await arkose_client.post(
                    url=self.ark0se_token_url,
                    json={"blob": arkose_dx, "method": arkose_method},
                    timeout=15
                )
                response_json = response.json()
                logger.info(f"Arkose token response: {response_json}")
                if response_json.get('solved', True):
                    self.ark0se_token = response_json.get('token')
                else:
                    raise HTTPException(status_code=403, detail="Failed to get Arkose token")
            except Exception:
                raise HTTPException(status_code=403, detail="Failed to get Arkose token")
            finally:
                await arkose_client.close()

    async def process_proof_of_work(self, proof_of_work: Dict[str, Any]):
        """
        Processes the proof of work challenge if required.

        :param proof_of_work: The proof of work information from the response.
        """
        if proof_of_work.get('required'):
            difficulty = proof_of_work.get("difficulty")
            if difficulty <= pow_difficulty:
                raise HTTPException(
                    status_code=403,
                    detail=f"Proof of work difficulty too high: {difficulty}"
                )
            seed = proof_of_work.get("seed")
            config = get_config(self.user_agent)
            self.proof_token, solved = await run_in_threadpool(
                get_answer_token, seed, difficulty, config
            )
            if not solved:
                raise HTTPException(status_code=403, detail="Failed to solve proof of work")

    async def handle_unsuccessful_chat_requirements(self, response):
        """
        Handles unsuccessful chat requirements responses.

        :param response: The response object.
        """
        if "application/json" == response.headers.get("Content-Type", ""):
            detail = response.json().get("detail", response.json())
        else:
            detail = await response.text()

        if "cf-please-wait" in detail:
            raise HTTPException(status_code=response.status_code, detail="cf-please-wait")
        if response.status_code == 429:
            raise HTTPException(status_code=response.status_code, detail="rate-limit")

        raise HTTPException(status_code=response.status_code, detail=detail)

    async def prepare_send_conversation(self) -> Dict[str, Any]:
        """
        Prepares the conversation request to be sent to the API.

        :return: The conversation request payload.
        """
        try:
            chat_messages, self.prompt_tokens = await api_messages_to_chat(
                self, self.api_messages, upload_by_url
            )
        except Exception as e:
            logger.error(f"Failed to format messages: {str(e)}")
            raise HTTPException(status_code=400, detail="Failed to format messages.")

        self.construct_chat_headers()
        self.construct_chat_request(chat_messages)
        return self.chat_request

    def construct_chat_headers(self):
        """
        Constructs the headers for the chat request.
        """
        self.chat_headers = self.base_headers.copy()
        self.chat_headers.update({
            'Accept': 'text/event-stream',
            'Openai-Sentinel-Chat-Requirements-Token': self.chat_token or "",
            'Openai-Sentinel-Proof-Token': self.proof_token or "",
        })
        if self.ark0se_token:
            self.chat_headers['Openai-Sentinel-Arkose-Token'] = self.ark0se_token

        if self.turnstile_token:
            self.chat_headers['Openai-Sentinel-Turnstile-Token'] = self.turnstile_token

        if conversation_only:
            keys_to_remove = [
                'Openai-Sentinel-Chat-Requirements-Token',
                'Openai-Sentinel-Proof-Token',
                'Openai-Sentinel-Arkose-Token',
                'Openai-Sentinel-Turnstile-Token'
            ]
            for key in keys_to_remove:
                self.chat_headers.pop(key, None)

    def construct_chat_request(self, chat_messages: List[Dict[str, Any]]):
        """
        Constructs the chat request payload.

        :param chat_messages: The list of chat messages.
        """
        conversation_mode = {"kind": "primary_assistant"}
        if "gpt-4-gizmo" in self.origin_model:
            gizmo_id = self.origin_model.split("gpt-4-gizmo-")[-1]
            conversation_mode = {"kind": "gizmo_interaction", "gizmo_id": gizmo_id}

        logger.info(f"Model mapping: {self.origin_model} -> {self.req_model}")

        self.chat_request = {
            "action": "next",
            "conversation_mode": conversation_mode,
            "force_nulligen": False,
            "force_paragen": False,
            "force_paragen_model_slug": "",
            "force_rate_limit": False,
            "force_use_sse": True,
            "history_and_training_disabled": self.history_disabled,
            "messages": chat_messages,
            "model": self.req_model,
            "parent_message_id": self.parent_message_id,
            "reset_rate_limits": False,
            "suggestions": [],
            "timezone_offset_min": -480,
            "variant_purpose": "comparison_implicit",
            "websocket_request_id": str(uuid.uuid4())
        }
        if self.conversation_id:
            self.chat_request['conversation_id'] = self.conversation_id

    async def send_conversation(self):
        """
        Sends the conversation to the API and processes the response.

        :return: The response from the API.
        """
        try:
            url = f'{self.base_url}/conversation'
            stream = self.data.get("stream", False)

            # Send the POST request with streaming
            response = await self.s.post_stream(
                url, headers=self.chat_headers, json=self.chat_request, timeout=10, stream=True
            )

            if response.status_code != 200:
                await self.handle_unsuccessful_conversation(response)

            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                res, start = await head_process_response(response.aiter_lines())
                if not start:
                    raise HTTPException(
                        status_code=403,
                        detail="Our systems have detected unusual activity coming from your system. Please try again later."
                    )
                if stream:
                    return stream_response(self, res, self.resp_model, self.max_tokens)
                else:
                    formatted_response = await format_not_stream_response(
                        stream_response(self, res, self.resp_model, self.max_tokens),
                        self.prompt_tokens,
                        self.max_tokens,
                        self.resp_model
                    )
                    return formatted_response
            elif "application/json" in content_type:
                response_text = await response.atext()
                resp_json = json.loads(response_text)
                raise HTTPException(status_code=response.status_code, detail=resp_json)
            else:
                response_text = await response.atext()
                raise HTTPException(status_code=response.status_code, detail=response_text)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def handle_unsuccessful_conversation(self, response):
        """
        Handles unsuccessful conversation responses.

        :param response: The response object.
        """
        response_text = await response.atext()
        if "application/json" == response.headers.get("Content-Type", ""):
            detail = json.loads(response_text).get("detail", json.loads(response_text))
            if response.status_code == 429:
                check_is_limit(detail, token=self.req_token, model=self.req_model)
        else:
            if "cf-please-wait" in response_text:
                raise HTTPException(status_code=response.status_code, detail="cf-please-wait")
            if response.status_code == 429:
                raise HTTPException(status_code=response.status_code, detail="rate-limit")
            detail = response_text[:100]

        raise HTTPException(status_code=response.status_code, detail=detail)

    async def get_download_url(self, file_id: str) -> str:
        """
        Retrieves the download URL for a given file ID.

        :param file_id: The ID of the file.
        :return: The download URL if successful, else an empty string.
        """
        url = f"{self.base_url}/files/{file_id}/download"
        headers = self.base_headers.copy()
        try:
            response = await self.s.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json().get('download_url', "")
            else:
                return ""
        except Exception:
            return ""

    async def get_download_url_from_upload(self, file_id: str) -> str:
        """
        Retrieves the download URL after uploading a file.

        :param file_id: The ID of the uploaded file.
        :return: The download URL if successful, else an empty string.
        """
        url = f"{self.base_url}/files/{file_id}/uploaded"
        headers = self.base_headers.copy()
        try:
            response = await self.s.post(url, headers=headers, json={}, timeout=5)
            if response.status_code == 200:
                return response.json().get('download_url', "")
            else:
                return ""
        except Exception:
            return ""

    async def get_upload_url(self, file_name: str, file_size: int, use_case: str = "multimodal") -> (str, str):
        """
        Retrieves the upload URL for a file.

        :param file_name: The name of the file.
        :param file_size: The size of the file in bytes.
        :param use_case: The use case for the file.
        :return: A tuple containing the file ID and upload URL.
        """
        url = f'{self.base_url}/files'
        headers = self.base_headers.copy()
        try:
            response = await self.s.post(
                url,
                headers=headers,
                json={
                    "file_name": file_name,
                    "file_size": file_size,
                    "timezone_offset_min": -480,
                    "use_case": use_case
                },
                timeout=5
            )
            if response.status_code == 200:
                res = response.json()
                file_id = res.get('file_id', "")
                upload_url = res.get('upload_url', "")
                logger.info(f"file_id: {file_id}, upload_url: {upload_url}")
                return file_id, upload_url
            else:
                return "", ""
        except Exception:
            return "", ""

    async def upload(self, upload_url: str, file_content: bytes, mime_type: str) -> bool:
        """
        Uploads a file to the given upload URL.

        :param upload_url: The URL to upload the file to.
        :param file_content: The content of the file.
        :param mime_type: The MIME type of the file.
        :return: True if successful, else False.
        """
        headers = self.base_headers.copy()
        headers.update({
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': mime_type,
            'X-Ms-Blob-Type': 'BlockBlob',
            'X-Ms-Version': '2020-04-08'
        })
        headers.pop('Authorization', None)
        try:
            response = await self.s.put(upload_url, headers=headers, data=file_content)
            return response.status_code == 201
        except Exception:
            return False

    async def upload_file(self, file_content: bytes, mime_type: str) -> Optional[Dict[str, Any]]:
        """
        Uploads a file to the API and returns the file metadata.

        :param file_content: The content of the file.
        :param mime_type: The MIME type of the file.
        :return: A dictionary containing file metadata if successful, else None.
        """
        if not file_content or not mime_type:
            return None

        width, height = None, None
        if mime_type.startswith("image/"):
            try:
                width, height = await get_image_size(file_content)
            except Exception as e:
                logger.error(f"Error processing image MIME type, changing to text/plain: {e}")
                mime_type = 'text/plain'

        file_size = len(file_content)
        file_extension = await get_file_extension(mime_type)
        file_name = f"{uuid.uuid4()}{file_extension}"
        use_case = await determine_file_use_case(mime_type)

        file_id, upload_url = await self.get_upload_url(file_name, file_size, use_case)
        if file_id and upload_url:
            if await self.upload(upload_url, file_content, mime_type):
                download_url = await self.get_download_url_from_upload(file_id)
                if download_url:
                    file_meta = {
                        "file_id": file_id,
                        "file_name": file_name,
                        "size_bytes": file_size,
                        "mime_type": mime_type,
                        "width": width,
                        "height": height,
                        "use_case": use_case
                    }
                    logger.info(f"File metadata: {file_meta}")
                    return file_meta
                else:
                    logger.error("Failed to get download URL")
            else:
                logger.error("Failed to upload file")
        else:
            logger.error("Failed to get upload URL")
        return None

    async def check_upload(self, file_id: str) -> bool:
        """
        Checks the status of an uploaded file.

        :param file_id: The ID of the file.
        :return: True if the file is ready, else False.
        """
        url = f'{self.base_url}/files/{file_id}'
        headers = self.base_headers.copy()
        try:
            for _ in range(30):
                response = await self.s.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    res = response.json()
                    retrieval_index_status = res.get('retrieval_index_status', '')
                    if retrieval_index_status == "success":
                        return True
                await asyncio.sleep(1)
            return False
        except Exception:
            return False

    async def get_response_file_url(self, conversation_id: str, message_id: str, sandbox_path: str) -> Optional[str]:
        """
        Retrieves the file URL from the response.

        :param conversation_id: The ID of the conversation.
        :param message_id: The ID of the message.
        :param sandbox_path: The sandbox path of the file.
        :return: The download URL if successful, else None.
        """
        try:
            url = f"{self.base_url}/conversation/{conversation_id}/interpreter/download"
            params = {
                "message_id": message_id,
                "sandbox_path": sandbox_path
            }
            headers = self.base_headers.copy()
            response = await self.s.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get("download_url")
            else:
                return None
        except Exception:
            logger.info("Failed to get response file URL")
            return None

    async def close_client(self):
        """
        Closes the HTTP client and any open connections.
        """
        if self.s:
            await self.s.close()
        if self.ws:
            await self.ws.close()
            del self.ws
