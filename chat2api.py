import asyncio
import types
import warnings
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from chatgpt.ChatService import ChatService
from chatgpt.authorization import token_service
import chatgpt.globals as globals
from chatgpt.reverseProxy import chatgpt_reverse_proxy
from utils.Logger import logger
from utils.config import api_prefix, scheduled_refresh
from utils.retry import async_retry

# Warning filters
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize the FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if scheduled_refresh:
    @app.on_event("startup")
    async def startup_event():
        loop = asyncio.get_event_loop()
        # Initialize the scheduler with the current event loop
        app.state.scheduler = AsyncIOScheduler(event_loop=loop)
        app.state.scheduler.add_job(
            id='refresh',
            func=token_service.refresh_all_tokensrefresh_all_tokens,
            trigger='cron',
            hour=3,
            minute=0,
            day='*/4',
            kwargs={'force_refresh': True}
        )
        app.state.scheduler.start()
        # Trigger the token refresh task on app startup
        loop.call_later(0, lambda: asyncio.create_task(token_service.refresh_all_tokens(force_refresh=False)))

    @app.on_event("shutdown")
    async def shutdown_event():
        # Retrieve the scheduler from the app state
        scheduler = app.state.scheduler
        scheduler.shutdown()

async def to_send_conversation(request_data, req_token):
    """Handles setting dynamic chat data and getting chat requirements."""
    chat_service = ChatService(req_token)
    try:
        await chat_service.set_dynamic_data(request_data)
        await chat_service.get_chat_requirements()
        return chat_service
    except HTTPException as e:
        await chat_service.close_client()
        raise
    except Exception as e:
        await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")

async def process_conversation(request_data, req_token):
    """Processes the chat service conversation."""
    chat_service = await to_send_conversation(request_data, req_token)
    await chat_service.prepare_send_conversation()
    res = await chat_service.send_conversation()
    return chat_service, res

@app.post(f"/{api_prefix}/v1/chat/completions" if api_prefix else "/v1/chat/completions")
async def send_conversation(request: Request, req_token: str = Depends(oauth2_scheme)):
    """Handles chat completions via POST request."""
    try:
        request_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid JSON body"})
    
    # Retry logic using async_retry
    chat_service, res = await async_retry(process_conversation, request_data, req_token)
    
    # Process the response and handle background tasks
    background = BackgroundTask(chat_service.close_client)
    if isinstance(res, types.AsyncGeneratorType):
        return StreamingResponse(res, media_type="text/event-stream", background=background)
    else:
        return JSONResponse(res, media_type="application/json", background=background)

@app.get(f"/{api_prefix}/tokens" if api_prefix else "/tokens", response_class=HTMLResponse)
async def upload_html(request: Request):
    """Serves the tokens count as an HTML response."""
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return templates.TemplateResponse("tokens.html", {"request": request, "api_prefix": api_prefix, "tokens_count": tokens_count})

def append_tokens_to_file(tokens, file_path="data/token.txt"):
    """Handles appending tokens to a file with better concurrency."""
    with open(file_path, "a", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")

@app.post(f"/{api_prefix}/tokens/upload" if api_prefix else "/tokens/upload")
async def upload_post(text: str = Form(...)):
    """Uploads and appends tokens to the token list and file."""
    lines = text.split("\n")
    new_tokens = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    
    if new_tokens:
        globals.token_list.extend(new_tokens)
        append_tokens_to_file(new_tokens)
    
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}

@app.post(f"/{api_prefix}/tokens/clear" if api_prefix else "/tokens/clear")
async def clear_tokens():
    """Clears all tokens and resets the token file."""
    globals.token_list.clear()
    globals.error_token_list.clear()
    
    # Clear the token file
    Path("data/token.txt").write_text("", encoding="utf-8")
    
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    return {"status": "success", "tokens_count": 0}

@app.post(f"/{api_prefix}/tokens/error" if api_prefix else "/tokens/error")
async def error_tokens():
    """Returns a list of error tokens."""
    error_tokens_list = list(set(globals.error_token_list))
    return {"status": "success", "error_tokens": error_tokens_list}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
async def reverse_proxy(request: Request, path: str):
    """Reverses proxy requests to the ChatGPT backend."""
    return await chatgpt_reverse_proxy(request, path)
