import os
import requests
import json
import yaml
from tavily import TavilyClient
from dotenv import load_dotenv

# ==========================================
# Dynamic Path Configuration
# ==========================================
# Calculate absolute paths for config and data directories to ensure the script works regardless of the current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

load_dotenv(dotenv_path=ENV_PATH)

os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# Global Configuration & YAML Loading
# ==========================================
WIKI_HEADERS = {
    "User-Agent": "WikiGenBench_Project/1.0 (itinglin1129@gmail.com) python-requests/2.31"
}
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ [Error] Failed to read YAML file: {e}")
else:
    print(f"⚠️ [Warning] Config file not found at exactly: {CONFIG_PATH}. Using defaults.")

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
        print("[Error] JSON parsing failed during the validation stage.")
        return False, None

    pages = response.get("query", {}).get("pages", {})
    
    for page_id, page_info in pages.items():
        if page_id == "-1":
            return False, None # Page does not exist at all
            
        # Check if we hit a disambiguation page
        if "pageprops" in page_info and "disambiguation" in page_info["pageprops"]:
            print(f"[Warning] '{user_input}' is a disambiguation page. Initiating automatic resolution...")
            
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
                print(f"[Success] System automatically resolved '{user_input}' to precise entity: '{exact_title}'")
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
        print(f"[Error] JSON parsing failed. Raw server response:\n{response.text}")
        return []
        
    sections = []
    
    # Exclude meta-sections not useful for RAG generation
    exclude_sections = [
        "see also", "references", "external links", "further reading", 
        "notes", "explanatory notes", "publications", "bibliography", 
        "gallery", "sources", "citations", "index", "notes and references", "footnotes", "howland_island"
    ]
    
    if "parse" in data and "sections" in data["parse"]:
        for sec in data["parse"]["sections"]:
            if sec["toclevel"] == 1:
                section_title = sec["line"]
                # Convert to lowercase for matching, filter out unwanted sections
                if section_title.lower() not in exclude_sections:
                    sections.append(section_title)
                    
    print(f"[Success] Retrieved clean Level-1 sections: {sections}")                
    return sections

# ==========================================
# Step 2.5: Text Chunking (Sliding Window)
# ==========================================
def chunk_text(text, chunk_size, overlap):
    """
    Splits long text into smaller chunks.
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        # Break early if we've reached the end of the text
        if i + chunk_size >= len(words):
            break
            
    return chunks

# ==========================================
# Step 3: Fetch Data via Tavily
# ==========================================
def fetch_tavily_rag_data(island_name, sections, api_key):
    """
    Fetches context data from Tavily, chunks long content, applies hard filters,
    and uses dynamic retrieval logic to ensure a high-quality baseline of chunks.
    """
    tavily_client = TavilyClient(api_key=api_key)
    
    # Extract parameters from YAML (with fallbacks based on your settings)
    chunk_size = CONFIG['retrieval']['chunking'].get('chunk_size', 500) if CONFIG else 500
    overlap = CONFIG['retrieval']['chunking'].get('overlap', 60) if CONFIG else 60
    
    # Dynamic Tavily settings
    max_results = CONFIG['retrieval']['tavily'].get('max_results', 10) if CONFIG else 10
    search_depth = CONFIG['retrieval']['tavily'].get('search_depth', "advanced") if CONFIG else "advanced"

    final_output = {
        "island_name": island_name,
        "sections_data": {}
    }
    
    chunk_counter = 1
    
    for section in sections:
        query = f"scientific facts and detailed analysis of {island_name} {section} -site:wikipedia.org"
        print(f"\n[Info] Searching via Tavily for: '{query}' (Depth: {search_depth}, Max Results: {max_results})")
        
        final_output["sections_data"][section] = {
            "chunks": []
        }
        
        try:
            # 1. Retrieval Phase 
            response = tavily_client.search(
                query=query, 
                search_depth=search_depth, 
                max_results=max_results, 
                include_answer=False
            )
            
            results = response.get("results", [])
            candidate_chunks = []
            
            # 2. Chunking & Hard Filtering Phase
            for res in results:
                url = res.get("url", "").lower()
                content = res.get("content", "")
                tavily_score = res.get("score", 0.0)

                # Hard Filter 1: Exclude Wikipedia, extremely short text (<100 chars), or link-heavy garbage
                if "wikipedia" in url or len(content) < 100 or content.count("http") > 5:
                    continue
                
                # Split the document into chunks using YAML parameters
                text_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
                
                for c_idx, chunk_str in enumerate(text_chunks):
                    # Hard Filter 2: Ensure the resulting chunk is not a tiny fragment (<50 chars)
                    if len(chunk_str) >= 50: 
                        candidate_chunks.append({
                            "text": chunk_str,
                            "source_url": res.get("url", ""),
                            "retrieval_score": tavily_score, 
                            "parent_doc_index": c_idx 
                        })

            # 3. Dynamic Retrieval Logic (Evaluation & Fallback)
            # Sort all valid candidates by their retrieval score in descending order
            candidate_chunks.sort(key=lambda x: x["retrieval_score"], reverse=True)
            
            # Identify high-quality chunks (Score > 0.65)
            high_quality_chunks = [c for c in candidate_chunks if c["retrieval_score"] > 0.65]
            
            # Selection Logic: If we have >= 5 high quality chunks, keep all of them.
            # Otherwise, apply the fallback mechanism and keep the top 5 highest scoring chunks overall.
            if len(high_quality_chunks) >= 5:
                selected_chunks = high_quality_chunks
                print(f"   -> [Filter] Kept {len(selected_chunks)} high-quality chunks (Score > 0.65).")
            else:
                selected_chunks = candidate_chunks[:5]
                print(f"   -> [Fallback] Insufficient high-quality data. Kept top {len(selected_chunks)} chunks overall.")

            # 4. Final Formatting & ID Assignment
            section_chunks = []
            for chunk in selected_chunks:
                chunk["chunk_id"] = f"chunk_{chunk_counter:04d}"
                section_chunks.append(chunk)
                chunk_counter += 1

            # Add the finalized chunks to the knowledge base
            final_output["sections_data"][section]["chunks"] = section_chunks
                
        except Exception as e:
            print(f"[Error] Exception occurred while processing '{query}': {e}")
            
    return final_output

# ==========================================
# Main Execution Pipeline 
# ==========================================
def run_rag_pipeline(target_island, tavily_api_key=None):
    """
    Executes the offline data retrieval pipeline to build the chunked knowledge base.
    """
    print(f"--- Starting Data Engineering pipeline for: {target_island} ---")
    
    # 1. Validation and Disambiguation
    is_valid, exact_island_name = validate_and_resolve_island_name(target_island)
    if not is_valid:
        print(f"[Error] Validation failed: '{target_island}' is not a valid Wikipedia entity.")
        return None
    
    print(f"[Process] Using precise entity name: {exact_island_name}")
    
    # 2. Retrieve Outline Structure
    sections = get_level_1_sections(exact_island_name)
    if not sections:
        print("[Warning] No sections found. Applying default blueprint...")
        sections = ["Geography", "Geology", "Ecology", "History"]
        
    # 3. Fetch and Chunk Context via Tavily
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
    
    # 5. Export to JSON file in the designated DATA directory
    safe_filename = exact_island_name.replace(" ", "_").replace("/", "_")
    output_filename = os.path.join(DATA_DIR, f"{safe_filename}_rag_context.json")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n[Success] Pipeline complete! Data exported to: {output_filename}")
    return final_output

# ==========================================
# Direct Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. 抓取 API Key
    MY_TAVILY_KEY = os.getenv("TAVILY_API_KEY")
    
    # 2. 如果沒抓到，印出詳細的錯誤訊息方便除錯
    if not MY_TAVILY_KEY:
        print(f"❌ [Fatal Error] TAVILY_API_KEY not found!")
        print(f"   程式試圖尋找的 .env 路徑是: {ENV_PATH}")
        exit(1)
        
    # 3. 從 YAML 讀取要測試的島嶼名稱
    TARGET_ENTITY = CONFIG.get("pipeline_config", {}).get("target_entity", "Nishinoshima")
    
    # 執行主程式
    pipeline_results = run_rag_pipeline(TARGET_ENTITY, MY_TAVILY_KEY)
