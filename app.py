import os
import json
from typing import List, Dict, Any
import math
import tiktoken
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st
import pinecone as pc
import pytz
import datetime
import base64
import requests
import openai
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
from pinecone import Pinecone
from credentials import openai_key, deepseek_key, pinecone_key, client_id, client_secret

# Initialize OpenAI client
openai.api_key = openai_key

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_key)

# API credentials
DEEPSEEK_API_KEY = deepseek_key
CLIENT_ID = client_id
CLIENT_SECRET = client_secret

# Configuration constants
REGION = "us-east"  # 8x8 API region
PAGE_SIZE = 10  # Results per page in UI
INDEX_NAME = "8x8-transcription-embeddings"  # Pinecone index name
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # DeepSeek API endpoint
DEEPSEEK_MODEL = "deepseek-chat"  # LLM model for analysis
MAX_TOKENS_PER_REQUEST = 60000  # Max tokens per LLM request
MAX_OUTPUT_TOKENS = 4096  # Max tokens in LLM response
TOKEN_SAFETY_BUFFER = 2000  # Token buffer to avoid exceeding limits

# Timezone setup for conversion
utc = pytz.utc
est = pytz.timezone("US/Eastern")

def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken encoding.
    Used to manage LLM context window limits.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_deepseek(prompt: str) -> str:
    """
    Query DeepSeek API with automatic retries on failure.
    Handles API communication and response parsing.
    """
    if not DEEPSEEK_API_KEY:
        st.error("DeepSeek API key missing!")
        return ""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"DeepSeek API error: {e}")
        return ""

def chunk_transcriptions(transcriptions: List[Dict[str, Any]], user_query: str) -> List[str]:
    """
    Split transcriptions into token-limited chunks for LLM processing.
    Formats chunks with query context and maintains conversation structure.
    """
    chunks = []
    current_chunk = []
    # Initialize with base tokens for query wrapper
    base_tokens = count_tokens(f"User query: {user_query}\n\nPlease analyze the following transcriptions:\n\n")
    current_token_count = base_tokens
    
    for idx, transcription in enumerate(transcriptions):
        # Format transcription with metadata
        formatted_transcription = (
            f"Transcription #{idx+1}\n"
            f"Doctor: {transcription.get('doctor_name', 'N/A')}\n"
            f"Patient: {transcription.get('patient_name', 'N/A')}\n"
            f"Time: {convert_to_est(transcription.get('startTime', 'N/A'))}\n"
            f"Conversation:\n{transcription.get('conversation', '')}\n\n"
        )
        
        transcription_tokens = count_tokens(formatted_transcription)
        
        # Start new chunk if current would exceed limit
        if current_token_count + transcription_tokens > MAX_TOKENS_PER_REQUEST - TOKEN_SAFETY_BUFFER:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [formatted_transcription]
            current_token_count = base_tokens + transcription_tokens
        else:
            current_chunk.append(formatted_transcription)
            current_token_count += transcription_tokens
    
    # Add final chunk if exists
    if current_chunk:
        chunks.append(current_chunk)
    
    # Build prompts for each chunk
    prompts = []
    for i, chunk in enumerate(chunks):
        chunk_text = "".join(chunk)
        prompt = (
            f"User query: {user_query}\n\n"
            f"Analyzing transcriptions (chunk {i+1}/{len(chunks)}):\n\n"
            f"{chunk_text}\n\n"
            f"Provide detailed response focusing on key insights and patterns."
        )
        prompts.append(prompt)
    
    return prompts

def process_with_deepseek(transcriptions: List[Dict[str, Any]], user_query: str) -> str:
    """
    Orchestrate multi-chunk analysis with DeepSeek.
    Handles chunking, progress display, and result aggregation.
    """
    if not transcriptions:
        return "No transcriptions to analyze."
    
    prompts = chunk_transcriptions(transcriptions, user_query)
    if not prompts:
        return "Error processing transcriptions."
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    chunk_results = []
    
    for i, prompt in enumerate(prompts):
        progress_value = (i) / len(prompts)
        progress_bar.progress(progress_value)
        status_text.text(f"Processing chunk {i+1}/{len(prompts)}...")
        chunk_results.append(query_deepseek(prompt))
    
    # Finalize UI
    progress_bar.progress(1.0)
    status_text.text("Finalizing results...")
    
    # Multi-chunk summarization
    if len(chunk_results) > 1:
        combined_results = "\n\n".join(
            [f"Chunk {i+1} results:\n{result}" for i, result in enumerate(chunk_results)]
        )
        final_prompt = (
            f"Original query: '{user_query}'\n\n"
            f"Combined analyses from {len(prompts)} chunks:\n\n{combined_results}\n\n"
            f"Synthesize comprehensive response highlighting key insights across all data."
        )
        final_result = query_deepseek(final_prompt)
        status_text.empty()
        return final_result
    
    status_text.empty()
    return chunk_results[0]

def convert_to_est(timestamp) -> str:
    """
    Convert various timestamp formats to EST string.
    Handles UNIX timestamps, ISO strings, and datetime objects.
    """
    if not timestamp or timestamp == "N/A":
        return "N/A"

    try:
        # Handle different input types
        if isinstance(timestamp, datetime):
            dt_utc = timestamp.replace(tzinfo=utc)
        elif isinstance(timestamp, (int, float)) or (isinstance(timestamp, str) and timestamp.isdigit()):
            # Convert from milliseconds to seconds
            dt_utc = datetime.utcfromtimestamp(float(timestamp)/1000).replace(tzinfo=utc)
        else:
            try:
                dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=utc)
            except ValueError:
                dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=utc)
                
        return dt_utc.astimezone(est).strftime("%Y-%m-%d %I:%M %p EST")
    except Exception as e:
        print(f"Timestamp conversion error: {timestamp}: {e}")
        return "N/A"

def generate_search_embedding(text: str) -> List[float]:
    """Generate text embedding using OpenAI's text-embedding-3-large model"""
    if not text or not isinstance(text, str):
        return []
    
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=text.lower(),
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return []

def convert_est_to_utc_timestamp(date) -> int:
    """Convert EST date to UTC timestamp in milliseconds for Pinecone filtering"""
    if not date:
        return None
    dt_est = datetime.combine(date, datetime.min.time()).replace(tzinfo=est)
    return int(dt_est.astimezone(utc).timestamp() * 1000)

def format_conversation(conversation_list) -> str:
    """Format conversation JSON into readable string with speaker labels"""
    return "\n\n".join(f'"{entry["speaker"]}":\n{entry["message"]}' for entry in conversation_list)

def vector_search(
    query_text: str, 
    start_date=None, 
    end_date=None, 
    direction=None, 
    include_doctor=True, 
    include_patient=True, 
    limit=50
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search with metadata filtering.
    Returns processed results with deduplication and structured data.
    """
    index = pc.Index(INDEX_NAME)
    filter_dict = {}
    
    # Date filtering
    if start_date:
        start_timestamp = convert_est_to_utc_timestamp(start_date)
        filter_dict.setdefault("startTime", {})["$gte"] = start_timestamp/1000
    
    if end_date:
        end_timestamp = convert_est_to_utc_timestamp(end_date)
        filter_dict.setdefault("startTime", {})["$lte"] = end_timestamp/1000
    
    # Name presence filters
    if include_doctor:
        filter_dict["doctor_name"] = {"$ne": "N/A"}
    if include_patient:
        filter_dict["patient_name"] = {"$ne": "N/A"}
    
    # Call direction filter
    if direction and direction.lower() != "all":
        filter_dict["direction"] = direction.upper()

    # Generate query embedding
    query_embedding = generate_search_embedding(query_text)
    if not query_embedding:
        return []
    
    # Configure search parameters
    search_params = {"top_k": limit, "include_metadata": True}
    if filter_dict:
        search_params["filter"] = filter_dict
    
    try:
        results = index.query(vector=query_embedding, **search_params)
        processed_results = []
        eight_x_ids = set()  # Track unique IDs for deduplication
        
        for match in results.matches:
            match_id = match.metadata.get("8x8_id", match.metadata.get("ID", match.id))
            
            # Skip duplicates
            if match_id in eight_x_ids:
                continue
            eight_x_ids.add(match_id)
            
            # Format conversation text
            conversation_text = ""
            if "conversation" in match.metadata and match.metadata["conversation"]:
                try:
                    conversation_text = format_conversation(json.loads(match.metadata["conversation"]))
                except:
                    conversation_text = match.metadata.get("transcription", "")
            elif "transcription" in match.metadata:
                conversation_text = match.metadata["transcription"]
            
            # Extract summary
            summary_field = match.metadata.get("summary", match.metadata.get("summery", ""))
            
            # Build result dictionary
            processed_results.append({
                "id": match_id,
                "score": match.score,
                "transcription": match.metadata.get("transcription", ""),
                "summary": summary_field,
                "doctor_name": match.metadata.get("doctor_name", ""),
                "patient_name": match.metadata.get("patient_name", ""),
                "email": match.metadata.get("email", ""),
                "startTime": match.metadata.get("startTime", ""),
                "duration": match.metadata.get("duration", ""),
                "direction": match.metadata.get("direction", ""),
                "type": match.metadata.get("type", ""),
                "conversation": conversation_text,
                "rightChannel": match.metadata.get("rightChannel", "N/A"),
                "leftChannel": match.metadata.get("leftChannel", "N/A"),
                "sentiment": match.metadata.get("sentiment", "N/A"),
                "userName": match.metadata.get("userName", "N/A"),
                "address": match.metadata.get("address", "N/A"),
                "extensionNumber": match.metadata.get("extensionNumber", "N/A")
            })
        
        return processed_results
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def get_access_token() -> str:
    """Obtain 8x8 API access token using client credentials"""
    if "access_token" not in st.session_state:
        credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {encoded_credentials}",
        }
        response = requests.post(
            "https://api.8x8.com/oauth/v2/token",
            headers=headers,
            data={"grant_type": "client_credentials"}
        )
        if response.status_code == 200:
            st.session_state.access_token = response.json()["access_token"]
        else:
            st.error("API token retrieval failed")
            return None
    return st.session_state.access_token

def get_recording(recording_id: str) -> bytes:
    """Fetch MP3 recording content from 8x8 storage API"""
    access_token = get_access_token()
    if not access_token:
        return None
        
    try:
        response = requests.get(
            f"https://api.8x8.com/storage/{REGION}/v3/objects/{recording_id}/content",
            headers={"Authorization": f"Bearer {access_token}"},
            stream=True
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Recording download failed: {e}")
        return None

# Extension to phone number mapping
extenstion_dict = {
    "1050": "+1 (301) 901-5667",
    "1004": "+1 (240) 387-5056",
    # ... (other extensions truncated for brevity)
    "1049": "+1 (240) 391-5787",
}

def request_recording(recording_id, unique_result_id):
    """
    Callback function for recording download.
    Stores audio in session state and keeps expander open.
    """
    with st.spinner("Fetching recording..."):
        audio_data = get_recording(recording_id)
        if audio_data:
            audio_storage_key = f"audio_{recording_id}_{unique_result_id}"
            st.session_state[audio_storage_key] = audio_data
            st.session_state[f"expander_{unique_result_id}_open"] = True

def main():
    """Main Streamlit application with search UI and results display"""
    st.set_page_config(page_title="Transcription Search", layout="wide")
    st.title("🔍 Advanced Transcription Search")
    st.markdown("Search through call transcriptions using semantic similarity and filters.")
    
    # Sidebar filters
    with st.sidebar:
        st.header("Search Filters")
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=None)
        end_date = st.date_input("End Date", value=None)
        
        st.subheader("Name Filters")
        include_doctor = st.checkbox("Include only calls with doctor name", value=True)
        include_patient = st.checkbox("Include only calls with patient name", value=True)

        st.subheader("Additional Filters")
        call_direction = st.selectbox("Call Direction", ["All", "Inbound", "Outbound"])
        top_records = st.number_input("Top numbers of records", 1, 6500, value=1000)
        
        if st.button("Reset Filters"):
            st.session_state.clear()
            st.rerun()
    
    # Main search UI
    search_query = st.text_input("Search transcriptions:", placeholder="Enter keywords or phrases...")
    col1, col2 = st.columns([1, 5])
    search_clicked = col1.button("Search", use_container_width=True)
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 0
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    # Execute search
    if search_clicked and search_query:
        with st.spinner("Searching transcriptions..."):
            direction_filter = None if call_direction == "All" else call_direction
            results = vector_search(
                search_query, 
                start_date, 
                end_date,
                direction=direction_filter,
                include_doctor=include_doctor,
                include_patient=include_patient,
                limit=top_records
            )
            st.session_state.search_results = results
            st.session_state.page = 0
            # Clear expander states
            for key in list(st.session_state.keys()):
                if key.startswith("expander_") and key.endswith("_open"):
                    st.session_state.pop(key)
    
    # Display results
    if st.session_state.search_results:
        total_results = len(st.session_state.search_results)
        st.success(f"Found {total_results} matching transcriptions")
        
        # DeepSeek analysis section
        st.markdown("## 🧠 DeepSeek LLM Analysis")
        with st.expander("Analyze transcriptions with DeepSeek AI", expanded=False):
            deepseek_query = st.text_area(
                "Analysis query:",
                placeholder="Example: Summarize common patient concerns",
                height=100
            )
            if st.button("Submit to DeepSeek", key="deepseek_submit"):
                total_tokens = sum(count_tokens(result.get("conversation", "")) 
                                for result in st.session_state.search_results)
                st.info(f"Estimated tokens: ~{total_tokens:,} (processed in chunks if needed)")
                
                with st.spinner(f"Analyzing {total_results} transcriptions..."):
                    analysis_result = process_with_deepseek(st.session_state.search_results, deepseek_query)
                    if analysis_result:
                        st.markdown("### Analysis Results")
                        st.markdown(analysis_result)
                        st.download_button(
                            label="Download Analysis",
                            data=analysis_result,
                            file_name="deepseek_analysis.txt",
                            mime="text/plain"
                        )
        
        # Pagination setup
        start_idx = st.session_state.page * PAGE_SIZE
        end_idx = min(start_idx + PAGE_SIZE, total_results)
        st.markdown("## 📋 Search Results")
        
        # Pagination controls
        col1, col2, col3 = st.columns([2, 4, 2])
        if col1.button("← Previous", disabled=(st.session_state.page == 0)):
            st.session_state.page -= 1
            st.rerun()
        col2.write(f"Page {st.session_state.page + 1} of {math.ceil(total_results / PAGE_SIZE)}")
        if col3.button("Next →", disabled=(end_idx >= total_results)):
            st.session_state.page += 1
            for key in list(st.session_state.keys()):
                if key.startswith("expander_") and key.endswith("_open"):
                    st.session_state.pop(key)
            st.rerun()
        
        # Display current page results
        current_page_results = st.session_state.search_results[start_idx:end_idx]
        for idx, result in enumerate(current_page_results, start=1):
            unique_result_id = f"{start_idx + idx}_{result.get('id', '')}"
            display_name = result.get('userName', 'Unknown')
            time_display = convert_to_est(result.get('startTime', 'N/A'))
            score_display = f"{result.get('score', 0):.2f}"
            
            # Manage expander state
            expander_id = f"expander_{unique_result_id}_open"
            is_expanded = st.session_state.get(expander_id, False)
            
            with st.expander(
                f"{start_idx + idx}. {display_name} - {time_display} (Score: {score_display})",
                expanded=is_expanded
            ):
                # Conversation display
                st.markdown("### Conversation")
                st.text_area(
                    "Conversation", 
                    value=result["conversation"], 
                    height=150, 
                    key=f"conv_{unique_result_id}",
                    label_visibility="collapsed"
                )
                
                # Metadata columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Call Details")
                    st.write(f"**Call ID:** {result.get('id', 'N/A')}")
                    st.write(f"**Doctor:** {result.get('doctor_name', 'N/A')}")
                    st.write(f"**Patient:** {result.get('patient_name', 'N/A')}")
                    st.write(f"**Direction:** {result.get('direction', 'N/A')}")
                    
                    # Handle number mapping based on call direction
                    if result.get('direction') == "OUTBOUND":
                        address = result.get('address', 'N/A')
                        st.write(f"**Call receiver:** {extenstion_dict.get(str(address), address)}")
                with col2:
                    st.markdown("### Additional Info")
                    st.write(f"**Start Time:** {time_display}")
                    if result.get('duration', 'N/A') != "N/A":
                        st.write(f"**Duration:** {int(result['duration'])/1000:.1f} seconds")
                    st.write(f"**Sentiment:** {result.get('sentiment', 'N/A')}")
                    st.write(f"**Username:** {result.get('userName', 'N/A')}")
                    
                    if result.get('direction') == "INBOUND":
                        address = result.get('address', 'N/A')
                        st.write(f"**Call sender:** {extenstion_dict.get(str(address), address)}")
                
                # Summary display
                if result.get("summary"):
                    st.markdown("### Summary")
                    st.text_area(
                        "Summary", 
                        value=result["summary"], 
                        height=100,
                        key=f"summ_{unique_result_id}",
                        label_visibility="collapsed"
                    )
                
                # Audio download section
                recording_id = result.get("id", "")
                dl_col1, dl_col2 = st.columns(2)
                request_key = f"req_{unique_result_id}"
                audio_storage_key = f"audio_{recording_id}_{unique_result_id}"
                
                # Request button
                if audio_storage_key not in st.session_state:
                    if dl_col1.button(
                        "Request Recording", 
                        key=request_key, 
                        on_click=request_recording, 
                        args=(recording_id, unique_result_id)
                    ):
                        pass  # Callback handles action
                
                # Download button
                if audio_storage_key in st.session_state:
                    dl_col2.download_button(
                        label="Download MP3",
                        data=st.session_state[audio_storage_key],
                        file_name=f"{recording_id}.mp3",
                        mime="audio/mpeg",
                        key=f"dl_{unique_result_id}"
                    )
    
    # No results message
    elif search_clicked and not st.session_state.search_results:
        st.warning("No matching transcriptions found. Try different search terms or filters.")

if __name__ == "__main__":
    main()