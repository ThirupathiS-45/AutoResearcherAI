# agents/search_agent.py

import requests
import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"

def search_arxiv(query, max_results=3):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_API_URL, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace).text.strip()
        summary = entry.find("atom:summary", namespace).text.strip()
        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib["href"]
        papers.append({
            "title": title,
            "summary": summary,
            "pdf_url": pdf_url
        })

    return papers
