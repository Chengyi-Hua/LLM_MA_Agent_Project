import os
import requests
import json
from tavily import TavilyClient

# ==========================================
# Global Configuration
# ==========================================
WIKI_HEADERS = {
    "User-Agent": "WikiGenBench_Project/1.0 (itinglin1129@gmail.com) python-requests/2.31"
}

# ==========================================
# Step 1: Validation and Disambiguation
# ==========================================
def validate_and_resolve_island_name(user_input):
    """
    1. Checks if the Wikipedia page exists.
    2. Checks if it is a 'disambiguation' page.
    3. If it is a disambiguation page, automatically appends 'island' to the search,
       and returns the precise entity name.
    """
    url = "https://en.wikipedia.org/w/api.php"
    
    params_check = {
        "action": "query",
        "format": "json",
        "titles": user_input,
        "prop": "pageprops", 
        "redirects": 1
    }
    
    try:
        response = requests.get(url, params=params_check, headers=WIKI_HEADERS).json()
    except requests.exceptions.JSONDecodeError:
        print("❌ JSON parsing failed (Validation stage).")
        return False, None

    pages = response.get("query", {}).get("pages", {})
    
    for page_id, page_info in pages.items():
        if page_id == "-1":
            return False, None # Page does not exist at all
            
        # Check if we hit a disambiguation page
        if "pageprops" in page_info and "disambiguation" in page_info["pageprops"]:
            print(f"⚠️ '{user_input}' is a disambiguation page. Initiating automatic resolution...")
            
            # Use Search API, silently appending "island" to narrow the scope
            search_query = f"{user_input} island"
            params_search = {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "format": "json",
                "srlimit": 1
            }
            search_res = requests.get(url, params=params_search, headers=WIKI_HEADERS).json()
            search_results = search_res.get("query", {}).get("search", [])
            
            if search_results:
                exact_title = search_results[0]["title"]
                print(f"✅ System automatically resolved '{user_input}' to precise entity: '{exact_title}'")
                return True, exact_title
            else:
                return False, None
        else:
            # If not disambiguation, return the original precise name
            return True, page_info["title"]

# ==========================================
# Step 2: Retrieve Level 1 Section Names
# ==========================================
def get_level_1_sections(island_name):
    """
    Retrieves the Level-1 section headings from Wikipedia to act as the blueprint.
    Filters out non-content sections.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "format": "json",
        "page": island_name,
        "prop": "sections",
        "redirects": 1
    }
    
    response = requests.get(url, params=params, headers=WIKI_HEADERS)
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"❌ JSON parsing failed. Raw server response:\n{response.text}")
        return []
        
    sections = []
    
    # [Extended Filter List] Exclude meta-sections not useful for RAG generation
    exclude_sections = [
        "see also", "references", "external links", "further reading", 
        "notes", "explanatory notes", "publications", "bibliography", 
        "gallery", "sources", "citations", "index"
    ]
    
    if "parse" in data and "sections" in data["parse"]:
        for sec in data["parse"]["sections"]:
            if sec["toclevel"] == 1:
                section_title = sec["line"]
                # Convert to lowercase for matching, filter out unwanted sections
                if section_title.lower() not in exclude_sections:
                    sections.append(section_title)
                    
    print(f"✅ Successfully retrieved clean Level-1 sections: {sections}")                
    return sections

# ==========================================
# Step 3: Fetch Data via Tavily
# ==========================================
def fetch_tavily_rag_data(island_name, sections, api_key):
    """
    Fetches context data from Tavily, organizes it hierarchically by section,
    filters out noise, and ensures a minimum number of fallback results.
    """
    tavily_client = TavilyClient(api_key=api_key)
    
    # Initialize the hierarchical output structure
    final_output = {
        "island_name": island_name,
        "sections_data": {}
    }
    
    chunk_counter = 1
    
    for section in sections:
        # Construct query, explicitly excluding Wikipedia to prevent data leakage
        query = f"scientific facts and detailed analysis of {island_name} {section} -site:wikipedia.org"
        print(f"🔍 Searching via Tavily for: '{query}'")
        
        # Initialize the data structure for the current section
        final_output["sections_data"][section] = {
            "chunks": []
        }
        
        try:
            # Perform advanced search to retrieve deep content
            response = tavily_client.search(
                query=query, 
                search_depth="advanced", 
                max_results=8, 
                include_answer=False
            )
            
            results = response.get("results", [])
            candidates = []
            
            for res in results:
                url = res.get("url", "").lower()
                content = res.get("content", "")
                score = res.get("score", 0.0)

                # Hard Filters: Skip Wikipedia, extremely short text, or link-heavy noise
                if "wikipedia" in url or len(content) < 100 or content.count("http") > 5:
                    continue
                
                # Append valid results to the candidates list
                candidates.append({
                    "text": content,
                    "source_url": res.get("url", ""),
                    "score": score
                })

            # Sort candidates by relevance score in descending order
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Retrieve high-quality chunks (score > 0.65)
            high_quality_chunks = [c for c in candidates if c['score'] > 0.65]
            
            # Selection Logic: Keep all high-quality chunks, or fallback to the top 5 if scarce
            selected_chunks = high_quality_chunks if len(high_quality_chunks) >= 5 else candidates[:5]

            # Format the selected chunks and assign unique IDs
            for chunk in selected_chunks:
                chunk_data = {
                    "chunk_id": f"chunk_{chunk_counter:03d}",
                    "score": chunk["score"], 
                    "text": chunk["text"],
                    "source_url": chunk["source_url"]
                }
                
                final_output["sections_data"][section]["chunks"].append(chunk_data)
                chunk_counter += 1
                
        except Exception as e:
            print(f"❌ Error occurred while searching for '{query}': {e}")
            
    return final_output

# ==========================================
# Main Execution Pipeline (Modularized)
# ==========================================
def run_rag_pipeline(target_island, tavily_api_key=None):
    """
    Executes the full data retrieval pipeline. 
    Returns a dictionary containing the formatted RAG context.
    """
    print(f"--- Starting entity processing for: {target_island} ---")
    
    # 1. Validation and Disambiguation
    is_valid, exact_island_name = validate_and_resolve_island_name(target_island)
    if not is_valid:
        print(f"⚠️ Validation failed: '{target_island}' is not a valid Wikipedia entity. Please try again.")
        return None
    
    print(f"🚀 Using precise entity name for subsequent retrieval: {exact_island_name}")
    
    # 2. Retrieve Outline Structure
    sections = get_level_1_sections(exact_island_name)
    
    # Fallback mechanism: Apply default blueprint if island is too new
    if not sections:
        print("⚠️ No sections found. Applying default volcanic island blueprint...")
        sections = ["Geography", "Geology", "Ecology", "History"]
        
    # 3. Fetch External Context via Tavily
    rag_data = fetch_tavily_rag_data(exact_island_name, sections, tavily_api_key)
    
    # 4. Build Final JSON Structure
    final_output = {
        "metadata": {
            "original_input": target_island,
            "resolved_entity_name": exact_island_name,
            "total_sections": len(sections)
        },
        "blueprint_data": rag_data
    }
    
    # 5. Export to JSON file (replace spaces/slashes to prevent file path errors)
    safe_filename = exact_island_name.replace(" ", "_").replace("/", "_")
    output_filename = f"{safe_filename}_rag_context.json"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"🎉 Processing complete! Data successfully exported to {output_filename}")
    
    # Return the dictionary so downstream agents can use it immediately in memory
    return final_output

# ==========================================
# Direct Execution Block
# ==========================================
if __name__ == "__main__":
    # If someone runs `python rag_data_pipeline.py` directly, this block executes.
    # Replace with your actual key or use os.getenv("TAVILY_API_KEY")
    MY_TAVILY_KEY = "please adjust to urs API Key" 
    TEST_ISLAND = "Nishinoshima"
    
    pipeline_results = run_rag_pipeline(TEST_ISLAND, MY_TAVILY_KEY)