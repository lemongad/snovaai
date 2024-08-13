import json
import os
from typing import List, Dict

import aiohttp
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
import asyncio
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
API_URL = os.getenv("API_URL", "https://fast.snova.ai/api/completion")
DEFAULT_MAX_TOKENS = 800
DEFAULT_STOP = ["<|eot_id|>"]

app = FastAPI()

async def event_stream(lines: List[str]):
    for line in lines:
        yield line + '\n\n'

async def call_api(messages: Dict, model: str) -> aiohttp.ClientResponse:
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
    }
    
    default_model = f"llama3-{model}"
    default_env_type = f'tp16{model}'

    data = {
        "body": {
            "messages": messages['messages'],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "stop": DEFAULT_STOP,
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
            "model": default_model
        },
        "env_type": default_env_type
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=data) as response:
            return await response.text()

@app.post("/v1/chat/completions")
async def completions(messages: Dict = Body(...)):
    logger.info(f'Received parameters: {messages}')

    model = messages.get('model', '405b')
    
    try:
        response_text = await call_api(messages, model)
        lines = response_text.splitlines()
        
        logger.info(f'API response: {response_text}')
        
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(event_stream(lines), headers=headers)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": "An internal error occurread"}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llama31_sambanova:app", host="0.0.0.0", port=8989, reload=True)
