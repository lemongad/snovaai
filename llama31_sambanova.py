import requests
import json
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import subprocess
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Completion(BaseModel):
    model: str = '405b'
    messages: list[object]
    stream: bool = None  # 默认为 None 表示未指定

@app.post("/v1/chat/completions")
async def completions(completion: Completion = Body(...)):
    print('接收参数：', completion)

    url = 'https://fast.snova.ai/api/completion'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
    }
    
    default_model = f"llama3-{completion.model}"
    default_env_type = f'tp16{completion.model}'

    data = {
        "body": {
            "messages": completion.messages,
            "stop": [
                "<|eot_id|>"
            ],
            "stream": True,  
            "stream_options": {
                "include_usage": True
            },
            "model": default_model
        },
        "env_type": default_env_type
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        response.raise_for_status()  # 检查是否返回HTTP错误

        # 如果stream参数未指定（None）或者为False，返回非流式
        if completion.stream:
            return StreamingResponse(response.iter_content(), media_type="text/event-stream")
        
        # 非流式处理
        lines = response.text.split('\n\n')
        
        new_lines = []
        for line in lines:
            if not line.startswith('data:'):
                continue
            line = line.removeprefix('data: ')
            if 'DONE' in line:
                continue
            j = json.loads(line)
            new_lines.append(j)
        
        # 获取数组最后一个对象
        last_obj = new_lines[-1]
        
        content = ''
        for line in new_lines:
            if len(line['choices']) == 0:
                continue
            delta = line['choices'][0]['delta']
            if 'content' not in delta:
                continue
            # 判断是否有content对象
            if 'content' in delta:
                content += line['choices'][0]['delta']['content']
        
        last_obj['choices'] = new_lines[0]['choices']
        last_obj['choices'][0]['delta']['content'] = content
        return last_obj

    except requests.RequestException as e:
        # 捕获所有请求相关的异常并返回适当的HTTP错误
        raise HTTPException(status_code=500, detail=f"API请求失败: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # 获取当前目录
    work_dir = os.path.dirname(os.path.abspath(__file__))
    # 进入工作目录
    os.chdir(work_dir)
    # 使用 subprocess 模块启动 Uvicorn
    subprocess.run(["uvicorn", "llama31_sambanova:app", "--reload", "--host", "0.0.0.0", "--port", "10000"])
