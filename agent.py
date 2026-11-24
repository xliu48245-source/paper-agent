"""
論文閱讀助理 v0.1 —— 手寫的 agent 迴圈。

執行：python agent.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

MAX_STEPS = 10  # 防止 agent 無限呼叫工具

SYSTEM_PROMPT = """你是一個論文研究助理，幫助研究生找論文、讀論文。

規則：
- 用繁體中文回答，但論文標題保留英文原文。
- 搜尋 arXiv 時使用英文關鍵字。
- 只根據工具回傳的內容描述論文，絕對不要編造論文或結果。
- 提到論文時附上 arXiv ID。
- 如果搜尋結果不相關，換個關鍵字再搜一次。
"""


# ---------- 1. 把工具說明轉成 Gemini 的格式 ----------

def build_gemini_tools() -> list[types.Tool]:
    declarations = [
        types.FunctionDeclaration(
            name=d["name"],
            description=d["description"],
            parameters_json_schema=d["parameters"],
        )
        for d in TOOL_DECLARATIONS
    ]
    return [types.Tool(function_declarations=declarations)]


# ---------- 2. 執行工具 ----------

def run_tool(name: str, args: dict) -> dict:
    """執行模型要求的工具。出錯時不讓程式崩潰，而是把錯誤回報給模型，讓它自己決定怎麼辦。"""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"未知的工具：{name}"}
    try:
        return {"result": func(**args)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ---------- 3. 呼叫模型（含 429 / 503 重試） ----------

def call_model(client, model, history, config, retries=3):
    for attempt in range(retries + 1):
        try:
            return client.models.generate_content(model=model, contents=history, config=config)
        except errors.APIError as e:
            if e.code in (429, 503) and attempt < retries:
                wait = 2 ** (attempt + 2)  # 4, 8, 16 秒
                print(f"  （API 忙碌或超過免費額度上限，{wait} 秒後重試…）")
                time.sleep(wait)
            else:
                raise


# ---------- 4. Agent 迴圈：這就是 agent 的核心 ----------

def agent_turn(client, model, config, history) -> str:
    for step in range(1, MAX_STEPS + 1):
        response = call_model(client, model, history, config)

        if not response.candidates:
            return "（模型沒有回應，可能被安全過濾擋下了）"

        # 把模型的回覆加進對話紀錄（包含它想呼叫的工具）
        history.append(response.candidates[0].content)

        # 模型沒有要呼叫工具 → 代表它認為可以回答了
        if not response.function_calls:
            return response.text or "（模型回傳了空白回答）"

        # 模型要呼叫工具 → 逐一執行，把結果放回對話紀錄
        result_parts = []
        for call in response.function_calls:
            args = dict(call.args or {})
            print(f"  [步驟 {step}] 呼叫 {call.name}({json.dumps(args, ensure_ascii=False)})")
            result = run_tool(call.name, args)
            if "error" in result:
                print(f"           ⚠ {result['error']}")
            result_parts.append(types.Part.from_function_response(name=call.name, response=result))

        history.append(types.Content(role="user", parts=result_parts))

    return f"（已達到 {MAX_STEPS} 步上限，agent 停止。可以試著把問題問得更具體。）"


# ---------- 5. 主程式：讀設定、開始對話 ----------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        sys.exit("請先在 .env 設定 GEMINI_API_KEY 和 GEMINI_MODEL（參考 .env.example）")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=build_gemini_tools(),
        # 關掉 SDK 的自動執行工具，我們要自己寫迴圈，才看得到每一步
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    history: list[types.Content] = []
    print(f"論文助理已啟動（模型：{model}）。輸入問題，輸入 exit 離開。")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        try:
            answer = agent_turn(client, model, config, history)
        except errors.APIError as e:
            answer = f"（API 錯誤 {e.code}：{e.message}）"
        print(f"\nAgent：{answer}")


if __name__ == "__main__":
    main()
