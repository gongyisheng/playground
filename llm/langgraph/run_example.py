# react_vendor_match.py
# pip install langgraph langchain_openai langchain-core

from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# 1) Tiny in-memory KB (mock)
# -----------------------------
VENDORS: List[Dict] = [
    {
        "vendor_id": "v_amazon",
        "vendor_name": "Amazon",
        "aliases": ["AMZN", "AMAZON MKTPLACE", "AMZN Mktp", "AMZN*"],
        "products": [
            {"product_id": "p_prime", "product_name": "Amazon Prime"},
            {"product_id": "p_marketplace", "product_name": "Marketplace Order"},
        ],
    },
    {
        "vendor_id": "v_apple",
        "vendor_name": "Apple",
        "aliases": ["APL*ITUNES", "APPLE.COM/BILL", "APPLE BILL", "ITUNES.COM/BILL"],
        "products": [
            {"product_id": "p_icloud", "product_name": "iCloud Storage"},
            {"product_id": "p_itunes", "product_name": "iTunes Purchase"},
        ],
    },
    {
        "vendor_id": "v_sptfy",
        "vendor_name": "Spotify",
        "aliases": ["SPOTIFY", "SPOTIFYUSA", "SPOTIFY*PREM"],
        "products": [
            {"product_id": "p_premium", "product_name": "Spotify Premium"},
        ],
    },
]

def _normalize(s: str) -> str:
    return " ".join(s.upper().split())

# -----------------------------
# 2) Example tools
# -----------------------------
@tool
def kb_search(query: str) -> List[Dict]:
    """Search the vendor KB by a short token (e.g., 'AMZN', 'APPLE BILL', 'SPOTIFY'). 
    Returns a list of candidate vendors with simple scores and matched aliases."""
    q = _normalize(query)
    results = []
    for v in VENDORS:
        score = 0
        matched_alias = None
        # direct vendor name hit
        if _normalize(v["vendor_name"]) in q:
            score += 3
            matched_alias = v["vendor_name"]
        # alias hits
        for al in v["aliases"]:
            if _normalize(al).rstrip("*") in q:
                score += 2
                matched_alias = al
        if score > 0:
            results.append({
                "vendor_id": v["vendor_id"],
                "vendor_name": v["vendor_name"],
                "matched_alias": matched_alias,
                "score": score,
                "products": v["products"],
            })
    # sort best-first
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

@tool
def alias_lookup(vendor_id: str) -> List[str]:
    """Return all known aliases for a given vendor_id."""
    for v in VENDORS:
        if v["vendor_id"] == vendor_id:
            return v["aliases"]
    return []

# -------------------------------------
# 3) ReAct-style prompt (system message)
# -------------------------------------
PROMPT = """You are VendorMatch ReAct, an expert vendor & product normalizer for messy bank transaction strings.
You have two tools: `kb_search` and `alias_lookup`. Follow this loop:

1) Retrieve: extract minimal, high-signal tokens (e.g., merchant stem like 'AMZN', 'APPLE BILL', 'SPOTIFY*PREM'). 
   Avoid sending the whole sentence unless necessary. Call `kb_search` with those tokens.
2) Reason: compare candidates by alias match quality, vendor name proximity, and context hints in the sentence
   (e.g., 'BILL', 'MKTPLACE', 'ITUNES').
3) Decide: pick the single best (vendor, product) if confidence is high. Prefer common subscription SKUs if hints
   exist (e.g., 'BILL' from Apple -> often iCloud/iTunes; 'Mktp' -> Amazon Marketplace).
4) Validate: if top-2 scores are close or evidence is weak, call `alias_lookup` to verify plausibility. If still ambiguous,
   explicitly mark `needs_review: true` and explain why.
5) Retrieve again if needed: you may issue another `kb_search` with a refined token.

Rules:
- Be conservative: never fabricate vendors outside the KB results.
- If nothing matches, return an explicit 'unknown' with `needs_review: true`.
- Final answer must be a JSON object only (no prose) with:
  {
    "vendor_id": str|null,
    "vendor_name": str|null,
    "product_id": str|null,
    "product_name": str|null,
    "confidence": float,          // 0-1
    "needs_review": boolean,
    "rationale": str,             // 1-2 concise sentences
    "candidates_considered": [    // up to 3, best-first
      {"vendor_id": str, "vendor_name": str, "score": number}
    ]
  }
Return ONLY that JSON in your final message.
"""

# -----------------------------
# 4) Build the agent
# -----------------------------
llm = ChatOpenAI(
    model="gpt-5-mini",
    reasoning={"effort": "minimal"},  # "minimal", "medium", "high"
    model_kwargs={"text": {"verbosity": "high"}},  # "low", "medium", or "high"
)  # or any tool-calling chat model you use
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[kb_search, alias_lookup],
    prompt=PROMPT,            # <-- here is the prompt you asked for
    # checkpointer=memory,      # optional, lets you keep short-term memory with thread_id
    debug=True
)

# -----------------------------
# 5) Example usage
# -----------------------------
if __name__ == "__main__":
    # Example transaction strings
    samples = [
        "POS 07/10 AMZN Mktp US*2J3KZ47Z AMZN.COM/BILL WA",
        "APPLE.COM/BILL 866-712-7753 CA 09/12",
        "SPOTIFY*PREM 09/01 866-123-4567 NY",
        "PAYMENT TO AMAZN (misspelled) 07/03",  # intentionally ambiguous/misspelled
    ]

    cfg = {"configurable": {"thread_id": "demo-1"}}  # enables memory across turns if you loop

    for s in samples:
        print("\n--- INPUT ---")
        print(s)
        result = agent.invoke({"messages": [HumanMessage(content=s)]}, cfg)
        final_msg = result["messages"][-1]
        print("\n--- JSON OUTPUT ---")
        print(final_msg.content)
