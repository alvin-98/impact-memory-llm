import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from google import genai
from pydantic import BaseModel
from typing import List, Callable
import io
from contextlib import redirect_stdout
import datetime
from ollama import chat, ChatResponse
import argparse
import time
import logging
from tqdm import tqdm
import json
import re
import textwrap

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY is None:
    raise ValueError("GEMINI_API_KEY not set in environment")

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")  # 'gemini' or 'ollama'
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

client = genai.Client(api_key=API_KEY)

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Rate limit controller (Option A: Token-bucket + backoff)
MAX_API_CALLS     = int(os.getenv("MAX_API_CALLS",     "1000"))
API_CALLS_PER_MIN = int(os.getenv("API_CALLS_PER_MIN", "60"))
API_STATE_FILE    = os.getenv("API_STATE_FILE", "api_state.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class APICallManager:
    def __init__(self, max_calls: int, per_minute: int = None):
        self.max_calls   = max_calls
        self.per_minute  = per_minute
        # load persistent state
        if os.path.exists(API_STATE_FILE):
            try:
                state = json.load(open(API_STATE_FILE))
                calls = state.get("calls", [])
                total = state.get("total_calls", 0)
            except Exception:
                calls, total = [], 0
        else:
            calls, total = [], 0
        now = time.time()
        # prune calls older than 60s
        self.calls       = [t for t in calls if now - t < 60]
        self.total_calls = total

    def wait_for_slot(self):
        now = time.time()
        # prune window
        self.calls = [t for t in self.calls if now - t < 60]
        # sleep if at per-minute cap
        if self.per_minute and len(self.calls) >= self.per_minute:
            wait = 60 - (now - self.calls[0])
            logger.warning(f"[Rate] sleeping {int(wait)}s…")
            time.sleep(wait + 0.1)
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
        # total quota check
        if self.total_calls >= self.max_calls:
            raise RuntimeError("API quota exhausted")
        # record this call
        self.calls.append(now)
        self.total_calls += 1
        # persist new state
        with open(API_STATE_FILE, "w") as f:
            json.dump({"calls": self.calls, "total_calls": self.total_calls}, f)

api_mgr = APICallManager(MAX_API_CALLS, API_CALLS_PER_MIN)

class CodeAgentResponse(BaseModel):
    function_name: str
    code: str

    @property
    def function(self) -> Callable:
        namespace: dict = {}
        exec(self.code, namespace)
        func = namespace.get(self.function_name)
        if func is None or not callable(func):
            raise ValueError(f"Function {self.function_name} not found after exec.")
        return func

class ReviewAgentResponse(BaseModel):
    test_cases: List[str]

def code_agent(requirement: str) -> CodeAgentResponse:
    if MODEL_BACKEND == "gemini":
        logger.info(f"Gemini code_agent call: '{requirement[:50]}...' (truncated)")
        api_mgr.wait_for_slot()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=requirement,
            config={
                "response_mime_type": "application/json",
                "response_schema": CodeAgentResponse,
            },
        )
        return response.parsed
    elif MODEL_BACKEND == "ollama":
        prompt = (
            requirement + "\n\nUse this JSON schema:\n"
            "CodeAgent = {'function_name': str, 'code': str}\nReturn: CodeAgent"
        )
        resp: ChatResponse = chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}])
        return CodeAgentResponse.parse_raw(resp.message.content)
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")

def review_agent(requirement: str, code: str) -> ReviewAgentResponse:
    if MODEL_BACKEND == "gemini":
        logger.info(f"Gemini review_agent call: '{requirement[:50]}...' (truncated)")
        api_mgr.wait_for_slot()
        prompt = (
            "You are a review agent. ONLY respond with JSON containing 'test_cases' as a list of Python assert statements to test the function. NO explanations or commentary."
            f"\nRequirement: {requirement}"
            f"\nFunction code:\n{code}"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": ReviewAgentResponse,
            },
        )
        return response.parsed
    elif MODEL_BACKEND == "ollama":
        prompt = (
            requirement + "\n\nUse this JSON schema:\n"
            "ReviewAgent = {'test_cases': list[str]}\nReturn: ReviewAgent"
        )
        resp: ChatResponse = chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":prompt}])
        return ReviewAgentResponse.parse_raw(resp.message.content)
    else:
        raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")

def run_review(requirement: str, code_resp: CodeAgentResponse) -> ReviewAgentResponse:
    review_resp = review_agent(requirement, code_resp.code)
    logger.info(f"Running {len(review_resp.test_cases)} review tests for function '{code_resp.function_name}'")
    results = []
    namespace = {code_resp.function_name: code_resp.function}
    for test in review_resp.test_cases:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(test, namespace)
            status, error = "pass", ""
        except AssertionError as e:
            status, error = "fail", str(e)
        except Exception as e:
            status, error = "error", str(e)
        results.append({
            "requirement": requirement,
            "function_name": code_resp.function_name,
            "test": test,
            "status": status,
            "error": error,
            "stdout": buf.getvalue(),
            "timestamp": datetime.datetime.now(),
        })
    df = pd.DataFrame(results)
    log_file = "review_logs.csv"
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)
    return review_resp

def orchestrator(eval_type: str, num_tasks: int = None):
    if eval_type == "mbpp":
        raw = load_dataset("Muennighoff/mbpp")["test"]
        df = pd.DataFrame(raw)
        get_tests = lambda r: r["test_list"]
        get_req = lambda r: r["text"]
        task_id_key = "task_id"
    else:
        raw = load_dataset("openai/openai_humaneval")["test"]
        df = pd.DataFrame(raw)
        get_tests = lambda r: r["test_list"]
        get_req = lambda r: r["prompt"]
        task_id_key = "task_id"

    # flatten DataFrame to list of dicts
    records = df.to_dict(orient="records")
    if num_tasks:
        records = records[:num_tasks]
    total_tasks = len(records)
    logger.info(f"Starting orchestrator: eval_type={eval_type}, tasks={total_tasks}")
    pm_code_agent_check_logs = []
    pm_review_agent_check_logs = []
    for idx, record in enumerate(tqdm(records, desc="Orchestrator", unit="task"), start=1):
        requirement = get_req(record)
        logger.info(f"[Orch] Task {idx}/{total_tasks}: '{requirement[:50]}...' (truncated)")
        code_resp = code_agent(requirement)
        review_resp = run_review(requirement, code_resp)
        fn = code_resp.function
        namespace = {code_resp.function_name: fn}
        ds_pass = ds_fail = 0
        for tc in get_tests(record):
            # patch test string to match generated function name
            m = re.match(r"\s*assert\s+([A-Za-z_]\w*)\s*\(", tc)
            if m:
                old = m.group(1)
                tc = tc.replace(old + "(", f"{code_resp.function_name}(")
                
            try:
                exec(tc, namespace)
                ds_pass += 1
            except:
                ds_fail += 1

        pm_code_agent_check_logs.append({
            "eval_type": eval_type,
            "task_id": record.get(task_id_key, idx),
            "requirement": requirement,
            "dataset_pass": ds_pass,
            "dataset_fail": ds_fail,
            "timestamp": datetime.datetime.now(),
        })

        # reference-code review-agent check
        if eval_type == "mbpp":
            ref_code = record.get("code", "")
        else:
            ref_code = record.get("prompt", "") + "\n" + record.get("canonical_solution", "")
        clean = textwrap.dedent(ref_code)
        ns_ref = {}
        exec(clean, ns_ref)
        # discover ref fn name
        for line in clean.splitlines():
            if line.strip().startswith("def "):
                ref_fn_name = line.strip().split()[1].split("(")[0]
                break
        else:
            raise ValueError("Could not parse ref fn name")
        ref_fn = ns_ref[ref_fn_name]
        review_pass = review_fail = 0
        for tc in review_resp.test_cases:
            parts = tc.split("(", 1)
            prefix = parts[0].strip()
            if prefix.startswith("assert "):
                old = prefix[len("assert "):]
                tc = tc.replace(old + "(", f"{ref_fn_name}(")
            try:
                exec(tc, {ref_fn_name: ref_fn})
                review_pass += 1
            except:
                review_fail += 1
        pm_review_agent_check_logs.append({
            "eval_type": eval_type,
            "task_id": record.get(task_id_key, idx),
            "review_pass": review_pass,
            "review_fail": review_fail,
            "timestamp": datetime.datetime.now(),
        })

    df_code = pd.DataFrame(pm_code_agent_check_logs)
    code_file = "pm_code_agent_check.csv"
    if os.path.exists(code_file):
        df_code.to_csv(code_file, mode="a", header=False, index=False)
    else:
        df_code.to_csv(code_file, index=False)
    print(f"Code-agent checks logged {len(pm_code_agent_check_logs)} tasks to {code_file}")

    # write review-agent check logs
    df_r = pd.DataFrame(pm_review_agent_check_logs)
    review_file = "pm_review_agent_check.csv"
    if os.path.exists(review_file):
        df_r.to_csv(review_file, mode="a", header=False, index=False)
    else:
        df_r.to_csv(review_file, index=False)
    print(f"Review-agent checks logged {len(pm_review_agent_check_logs)} tasks to {review_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_type", choices=["mbpp","humaneval"], required=True)
    parser.add_argument("--num_tasks", type=int, default=None)
    args = parser.parse_args()
    orchestrator(args.eval_type, args.num_tasks)

if __name__ == "__main__":
    main()