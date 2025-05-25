from duckduckgo_search import DDGS

def live_internet_search_duckduckgo(query):
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, region="in-en", safesearch="moderate", max_results=5):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })

    return results
