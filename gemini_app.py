import streamlit as st
import google.generativeai as genai
import os
import time
import json
import asyncio
import aiohttp
import pandas as pd
import datetime
from datetime import datetime, timedelta
import requests
from concurrent.futures import ThreadPoolExecutor
from credentials import google_key

# Configure page
st.set_page_config(page_title="Dental Lab Call Analysis", page_icon="🦷", layout="wide")

# Initialize APIs
def initialize_apis():
    # Initialize Gemini API
    google_api_key = google_key
    
    if not google_api_key:
        st.error("Google API key not found. Please set the required environment variable.")
        st.stop()
    
    # Initialize Gemini
    genai.configure(api_key=google_api_key)
    # model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17")
    return model

# Load the JSON file URLs from environment or config
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_json_file_urls():
    """
    Get the list of JSON file URLs either from environment variables or a config file.
    In a production setting, this could be dynamically fetched from Firebase.
    """
    # For demo purposes, we'll use environment variables
    # In production, you might want to fetch this list from Firebase
    json_urls_str = os.getenv("JSON_FILE_URLS", "")
    
    if not json_urls_str:
        # Fallback to some example URLs for demonstration
        return [
            "https://example.com/dental_transcripts_jan2025.json",
            "https://example.com/dental_transcripts_feb2025.json",
            "https://example.com/dental_transcripts_mar2025.json",
            "https://example.com/dental_transcripts_apr2025.json"
        ]
    
    return json_urls_str.split(",")

# Fetch JSON data from a URL
async def fetch_json_data(url, session):
    """Fetch JSON data from a URL asynchronously"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                st.error(f"Error fetching data from {url}: HTTP {response.status}")
                return []
    except Exception as e:
        st.error(f"Error fetching data from {url}: {str(e)}")
        return []

# Filter transcriptions by date range
def filter_transcriptions_by_date(transcriptions, start_timestamp, end_timestamp):
    """Filter transcriptions by start time within the given timestamp range"""
    # if start_timestamp= 
    return [t for t in transcriptions if start_timestamp <= t.get("startTime", 0) <= end_timestamp]

# Process chunks with Gemini
async def process_chunk_async(model, query, transcriptions, session):
    """Process a single chunk of transcriptions with Gemini asynchronously"""
    try:
        # Extract just the transcription text to save tokens
        transcription_texts = [t.get("transcription", "") for t in transcriptions]
        
        # Join with separators
        chunk_text = "\n\n--NEXT CALL--\n\n".join(transcription_texts)
        
        prompt = f"""
        You are a dental lab business analyst who specializes in analyzing call data for dental labs.
        Analyze these call transcriptions and answer this question: {query}
        
        TRANSCRIPTIONS:
        {chunk_text}
        
        Provide a clear, business-oriented answer that:
        1. Directly addresses the question with specific numbers and percentages
        2. Uses dental lab terminology that a lab manager would understand
        3. NEVER mentions that this is just part of the data or one "chunk" or "batch"
        4. Presents your findings as if they are complete (do not mention that this is partial data)
        5. Uses simple business language
        6. Avoids unnecessarily technical terms
        
        Keep your response concise and focused on business insights that a dental lab owner would find valuable.
        """
        
        response = await model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        # Return a generic error that doesn't mention chunks or batches
        return "Some call data couldn't be fully analyzed. The overall findings are based on the successfully analyzed calls."
    
    

# Create chunks from transcription data
def create_chunks(transcriptions, max_tokens_per_chunk=900000):
    """
    Split transcriptions into chunks that fit within token limits
    This is a simplified version - in production you'd need more precise token counting
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    avg_tokens_per_char = 0.25  # Rough estimate of tokens per character
    
    for transcript in transcriptions:
        # Estimate tokens in this transcript
        transcript_text = transcript.get("transcription", "")
        transcript_tokens = len(transcript_text) * avg_tokens_per_char
        
        # If adding this transcript would exceed the chunk limit, start a new chunk
        if current_tokens + transcript_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [transcript]
            current_tokens = transcript_tokens
        else:
            # Add to current chunk
            current_chunk.append(transcript)
            current_tokens += transcript_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Modify the aggregate_results function to make outputs more business-friendly
def aggregate_results(chunk_results, query):
    """
    Combine results from multiple chunks into a single coherent answer
    that's suitable for non-technical business users, with no technical references
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-pro-preview-03-25")
        
        prompt = f"""
        You are a dental lab business analyst presenting findings to a non-technical dental lab executive.
        Below are multiple analysis results from different sets of call transcriptions, 
        all answering this question: 
        
        QUESTION: {query}
        
        ANALYSIS RESULTS FROM DIFFERENT SETS:
        {chunk_results}
        
        Create a clear, concise executive summary that:
        1. Starts with the direct answer to the question in 1-2 sentences using rounded numbers
        2. Provides only the most important supporting data (no more than 3 key points)
        3. Uses simple business language a dental lab manager would understand
        4. NEVER mentions technical terms like "chunks", "batches", "sets", "analysis", etc.
        5. NEVER mentions how the data was processed, split, or analyzed
        6. NEVER references rate limits, processing errors, or technical difficulties
        7. If there are different percentages or numbers from different parts of the data, simply provide 
           an average or range without explaining where the different numbers came from
        8. Limits the entire response to 3-4 short paragraphs maximum
        9. Concludes with 1 actionable recommendation if appropriate
        
        Your response should read like a simple business insight with no hint of the technical 
        processing behind it. The dental lab manager should never know this came from an AI 
        or that the data was processed in multiple parts.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error processing results: {e}")
        return "We couldn't complete the analysis. Please try again or contact technical support."

async def process_query(query, model, start_date, end_date):
    with st.status("Analyzing your call data...", expanded=True) as status:
        try:
            # Convert dates to Unix timestamps
            start_timestamp = None
            end_timestamp = None
            if start_date:
                start_timestamp = datetime.combine(start_date, datetime.min.time()).timestamp()
            if end_date:
                end_timestamp = datetime.combine(end_date, datetime.max.time()).timestamp()
            
            # Load data
            with open("Chat_bot_Data.json", "r") as json_file:
                all_transcriptions = json.load(json_file)

            # Filter by date range
            status.update(label="Finding relevant calls...")
            if start_timestamp is not None and end_timestamp is not None:
                filtered_transcriptions = [t for t in all_transcriptions if start_timestamp <= t.get("startTime", 0) <= end_timestamp]
            elif start_timestamp is not None:
                filtered_transcriptions = [t for t in all_transcriptions if start_timestamp <= t.get("startTime", 0)]
            elif end_timestamp is not None:
                filtered_transcriptions = [t for t in all_transcriptions if t.get("startTime", 0) <= end_timestamp]
            else:
                filtered_transcriptions = all_transcriptions

            if not filtered_transcriptions:
                status.update(label="No calls found", state="error")
                return "No calls found in the selected date range. Please try expanding your date range."
                
            call_count = len(filtered_transcriptions)
            status.update(label=f"Analyzing {call_count} calls...")
            
            # Create chunks of appropriate size
            chunks = create_chunks(filtered_transcriptions)
            
            # Process chunks with rate limiting
            chunk_results = []
            total_chunks = len(chunks)
            
            # Use a simple progress bar instead of detailed updates
            progress_bar = st.progress(0)
            
            async with aiohttp.ClientSession() as session:
                for i, chunk_batch in enumerate(range(0, len(chunks), 4)):  # Process up to 4 chunks per minute
                    # Process a batch of chunks
                    batch = chunks[chunk_batch:chunk_batch+4]
                    tasks = [process_chunk_async(model, query, chunk, session) for chunk in batch]
                    
                    batch_results = await asyncio.gather(*tasks)
                    chunk_results.extend(batch_results)
                    
                    # Update progress
                    progress = min(1.0, (i + 1) * 4 / total_chunks)
                    progress_bar.progress(progress)
                    
                    # Generic status update that doesn't reveal technical details
                    status.update(label="Analyzing calls...")
                    
                    # Rate limiting - wait if needed
                    if chunk_batch + 4 < len(chunks):
                        await asyncio.sleep(60)  # Simple delay without explanation
            
            # Aggregate results
            status.update(label="Preparing your insights...")
            
            # Join results but remove any batch numbering
            all_results = "\n\n".join([result for result in chunk_results])
            final_answer = aggregate_results(all_results, query)
            
            # Remove progress bar
            progress_bar.empty()
            
            status.update(label="Analysis Complete", state="complete")
            return final_answer
            
        except Exception as e:
            status.update(label="Error in analysis", state="error")
            return "We encountered a problem analyzing your call data. Please try again or contact technical support."
    
# Streamlit UI
def main():
    st.title("🦷 Dental Lab Call Insights")
    st.write("Get quick answers from your call recordings with a simple question.")
    
    # Initialize APIs
    try:
        model = initialize_apis()
    except Exception as e:
        st.error("Our system is currently unavailable. Please try again later.")
        st.stop()
    
    # Date range selector in a card-like container
    st.subheader("1. Select Date Range (Optional)")
    date_container = st.container(border=True)
    with date_container:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", value=None)
        with col2:
            end_date = st.date_input("To", value=None)
        
        # Verify date range is valid
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                st.error("Error: Start date must be before end date")
    
    # Question section
    st.subheader("2. Ask Your Question")
    question_container = st.container(border=True)
    with question_container:
        # Examples of questions
        st.caption("Try one of these questions or write your own:")
        
        example_questions = [
            "What percentage of clients were disappointed?",
            "What are the top reasons for customer complaints?",
            "How many calls mentioned shipping delays?",
            "What percentage of orders were canceled?",
            "What do clients like most about our service?"
        ]
        
        # Create clickable examples in two rows
        cols1 = st.columns(3)
        cols2 = st.columns(2)
        
        # First row
        for i in range(3):
            if cols1[i].button(example_questions[i], key=f"example_{i}", use_container_width=True):
                st.session_state.query = example_questions[i]
        
        # Second row
        for i in range(3, 5):
            idx = i - 3
            if cols2[idx].button(example_questions[i], key=f"example_{i}", use_container_width=True):
                st.session_state.query = example_questions[i]
        
        # User input with example text
        if 'query' not in st.session_state:
            st.session_state.query = ""
        
        st.write("")  # Add some spacing
        query = st.text_input(
            "Your question:",
            value=st.session_state.query,
            placeholder="Example: What percentage of clients were disappointed?"
        )
        
        # Submit button
        submit_col1, submit_col2 = st.columns([1, 4])
        with submit_col1:
            submit_button = st.button("Get Answer", type="primary", use_container_width=True)
    
    # Results section
    if submit_button:
        if not query:
            st.warning("Please enter a question.")
        else:
            # Add a spinner with a more business-friendly message
            with st.spinner("Finding answers in your call data..."):
                # Process query
                result = asyncio.run(process_query(query, model, start_date, end_date))
                
                # Display result in a nice card-like container
                result_container = st.container(border=True)
                with result_container:
                    st.subheader("📊 Key Findings")
                    st.markdown(result)
                    
                    # Add option to download as PDF (placeholder)
                    st.download_button(
                        "Download as Report", 
                        result, 
                        "dental_lab_analysis.txt",
                        help="Download this analysis as a text file"
                    )
    
    # Add helpful information in an expander at the bottom
    with st.expander("Tips for better results"):
        st.markdown("""
        - **Be specific** in your questions (e.g., "What percentage of calls mentioned shipping delays?" instead of "Tell me about shipping")
        - **Use the date range** to focus on specific time periods
        - **Ask one question at a time** for clearer answers
        - For complex questions, break them down into smaller, more specific questions
        """)

if __name__ == "__main__":
    main()