"""
Agent 可以使用的工具。

每個工具由兩部分組成：
1. 一個普通的 Python 函式（真正做事的程式）
2. 一份「工具說明」（給模型看的 JSON Schema，告訴模型這個工具做什麼、要傳什麼參數）

工具說明刻意寫成純 dict，不綁定任何一家的 SDK，
之後要換成 Claude 或 OpenAI，只要在 agent.py 轉換格式即可。
"""

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


# ---------- 內部輔助函式 ----------

def _fetch_arxiv(params: dict) -> bytes:
    """呼叫 arXiv API，回傳原始 XML。"""
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "paper-agent-practice/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _parse_entries(xml_bytes: bytes) -> list[dict]:
    """把 arXiv 回傳的 Atom XML 轉成 list of dict。"""
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall("atom:entry", NS):

        def text(tag: str) -> str:
            el = entry.find(tag, NS)
            # arXiv 的文字常有多餘換行和空白，壓成一行
            return " ".join(el.text.split()) if el is not None and el.text else ""

        arxiv_id = text("atom:id").rsplit("/abs/", 1)[-1]
        authors = [text_el.text.strip()
                   for a in entry.findall("atom:author", NS)
                   if (text_el := a.find("atom:name", NS)) is not None and text_el.text]
        papers.append({
            "id": arxiv_id,
            "title": text("atom:title"),
            "authors": authors,
            "published": text("atom:published")[:10],
            "abstract": text("atom:summary"),
        })
    return papers


# ---------- 工具本體 ----------

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """搜尋 arXiv。為了省 token，摘要只回傳前 300 字。"""
    max_results = max(1, min(int(max_results), 10))  # 不信任模型給的數字，自己限制範圍
    xml_bytes = _fetch_arxiv({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    papers = _parse_entries(xml_bytes)
    for p in papers:
        if len(p["abstract"]) > 300:
            p["abstract"] = p["abstract"][:300] + "..."
        if len(p["authors"]) > 3:
            p["authors"] = p["authors"][:3] + ["et al."]
    return papers


def get_paper(arxiv_id: str) -> dict:
    """用 arXiv ID 取得單篇論文的完整摘要。"""
    papers = _parse_entries(_fetch_arxiv({"id_list": arxiv_id.strip()}))
    if not papers or not papers[0]["title"]:
        return {"error": f"找不到論文 {arxiv_id}"}
    return papers[0]


# ---------- 給模型看的工具說明 ----------

TOOL_DECLARATIONS = [
    {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for academic papers by keywords. Returns a list of papers "
            "with id, title, authors, publish date and a truncated abstract. "
            "Use English keywords for best results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords in English, e.g. 'LLM hallucination detection'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return, 1-10. Default 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_paper",
        "description": (
            "Get the full abstract and metadata of one arXiv paper by its ID. "
            "Use this after search_arxiv when you need the complete abstract."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID, e.g. '2310.06825v1'",
                },
            },
            "required": ["arxiv_id"],
        },
    },
]

# 工具名稱 → 實際要執行的函式
TOOL_FUNCTIONS = {
    "search_arxiv": search_arxiv,
    "get_paper": get_paper,
}
