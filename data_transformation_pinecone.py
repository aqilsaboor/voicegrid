import os
import zipfile
import json
import shutil
import time
from datetime import datetime
from typing import List
from trancribe import trancribe
from pinecone import Pinecone, ServerlessSpec
from LLM_Models import DeepSeek, verify_with_gpt4
import openai
from credentials import openai_key, pinecone_key

# Initialize OpenAI with API key
openai.api_key = openai_key

# Initialize Pinecone client for embeddings storage
pc = Pinecone(api_key=pinecone_key, environment="us-east-1")
index_name = "8x8-transcription-embeddings"

# Create index if it doesn't exist, specifying vector dimension and metric
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
# Connect to the index
index = pc.Index(index_name)


def data_transformation(zipfile_name: str, json_file_name: str, day: str = None):
    """
    Unzip call recording files, parse metadata, transcribe audio,
    extract insights via DeepSeek and GPT-4, and upsert embeddings to Pinecone.

    Args:
        zipfile_name: Name of the ZIP archive with audio files.
        json_file_name: Name of the JSON metadata file.
        day: Optional description of the time window.
    """
    # Create a folder based on the ZIP filename and extract contents
    folder_name = zipfile_name.replace(".zip", "")
    with zipfile.ZipFile(zipfile_name, 'r') as zip_ref:
        zip_ref.extractall(folder_name)
    # Remove the zip after extraction to save space
    os.remove(zipfile_name)

    # Load the call metadata
    with open(json_file_name, "r") as f:
        call_data = json.load(f)

    def parse_datetime(date_str: str) -> int:
        """
        Convert ISO datetime strings to UNIX timestamp in milliseconds.
        Supports multiple string formats.
        """
        date_formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        print(f"Skipping invalid date format: {date_str}")
        return date_str

    def parse_timestamp(ts) -> int:
        """Ensure numeric timestamp is int milliseconds."""
        try:
            return int(ts)
        except ValueError:
            print(f"Skipping invalid timestamp: {ts}")
            return ts

    def convert_record(record: dict) -> dict:
        """
        Normalize metadata: parse dates and flatten tags list to dict.
        """
        # Parse main date fields
        for field in ("createdTime", "updatedTime"):
            if field in record and isinstance(record[field], str):
                record[field] = parse_datetime(record[field])
        # Convert list of tags to dictionary for easy access
        record["tags"] = {t["key"]: t["value"] for t in record.get("tags", [])}
        # Parse nested timestamps
        for tfield in ("startTime", "endTime"):
            if tfield in record["tags"]:
                record["tags"][tfield] = parse_timestamp(record["tags"][tfield])
        return record

    def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
        """
        Request embeddings from OpenAI in batch for a list of texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        if not valid_texts:
            return []
        try:
            resp = openai.embeddings.create(
                model="text-embedding-3-large",
                input=valid_texts,
                encoding_format="float"
            )
            return [data.embedding for data in resp.data]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    # Process each recording in the metadata
    for idx, rec in enumerate(call_data, start=1):
        rec_id = rec.get("id")
        # Check if this, or its summary, already exists in Pinecone
        try:
            existing = index.query(id=rec_id, top_k=1)
        except Exception:
            time.sleep(5)
            existing = index.query(id=rec_id, top_k=1)
        if existing.get('matches'):
            print(f"{idx}/{len(call_data)}: ID {rec_id} already indexed.")
            continue

        print(f"{idx}/{len(call_data)}: Processing ID {rec_id} ({day})")
        # Normalize the metadata fields
        rec = convert_record(rec)
        # Build file path and ensure startTime is in seconds
        fname = rec.get("objectName", "").replace(':', '.')
        audio_path = os.path.join(folder_name, fname)
        if len(str(rec['tags']['startTime'])) > 10:
            rec['tags']['startTime'] /= 1000

        # Construct base metadata dict for Pinecone
        metadata = {
            '8x8_id': rec_id,
            'direction': rec['tags'].get('direction'),
            'duration': rec['tags'].get('duration'),
            'leftChannel': rec['tags'].get('leftChannelEndpointName1', 'N/A'),
            'rightChannel': rec['tags'].get('rightChannelEndpointName1', 'N/A'),
            'startTime': rec['tags'].get('startTime'),
            'userName': rec['tags'].get('userName'),
            'type': rec.get('type'),
            'address': rec['tags'].get('address'),
            'extensionNumber': rec['tags'].get('extensionNumber')
        }
        # Transcribe audio and add to metadata
        transcription = trancribe(audio_path)
        metadata['transcription'] = transcription

        # Only proceed if transcription is substantial
        if len(transcription) > 50:
            # Extract details via DeepSeek and validate with GPT-4
            deepseek_data = DeepSeek(transcription)
            verified = verify_with_gpt4(deepseek_data, transcription)
            # Add conversational fields
            metadata.update({
                'conversation': json.dumps(verified['Conversation']),
                'doctor_name': verified.get('Doctor Name'),
                'patient_name': verified.get('Patient Name'),
                'email': verified.get('Email'),
                'sentiment': verified.get('Sentiment'),
                'summary': verified.get('Summary', verified.get('Summery', 'N/A'))
            })

            # Embed transcription text
            emb = generate_embeddings_batch([transcription.lower()])
            if emb and emb[0]:
                index.upsert([{
                    'id': rec_id,
                    'values': emb[0],
                    'metadata': metadata
                }])
            # Embed summary text if available
            if metadata['summary'] != 'N/A':
                emb2 = generate_embeddings_batch([metadata['summary'].lower()])
                if emb2 and emb2[0]:
                    index.upsert([{
                        'id': f"{rec_id}-summary",
                        'values': emb2[0],
                        'metadata': metadata
                    }])
            print(f"Record {rec_id} added to Pinecone.")

    # Clean up extracted folder
    shutil.rmtree(folder_name)
    print(f"Cleaned up folder: {folder_name}")


def data_update(json_file_name: str, day: str = None):
    """
    Update existing Pinecone vectors with new metadata fields (address & extension).
    """
    # Load metadata
    with open(json_file_name, 'r') as f:
        call_data = json.load(f)
    
    def parse_datetime(date_str):
        """Convert various datetime string formats to integer timestamp (ms)."""
        date_formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return int(dt.timestamp() * 1000)  # Convert to milliseconds
            except ValueError:
                continue
        print(f"Skipping invalid date format: {date_str}")
        return date_str  # Return original string if parsing fails

    def convert_record(record: dict) -> dict:
        # Same date and tag parsing logic as above
        for field in ("createdTime", "updatedTime"):
            if field in record and isinstance(record[field], str):
                record[field] = parse_datetime(record[field])
        if isinstance(record.get('tags'), list):
            record['tags'] = {t['key']: t['value'] for t in record['tags']}
        return record

    # Iterate and update metadata for each existing vector
    for idx, rec in enumerate(call_data, start=1):
        rec_id = rec.get('id')
        # Check base vector exists, then update
        result = index.query(id=rec_id, top_k=1)
        if result.get('matches'):
            rec = convert_record(rec)
            index.update(
                id=rec_id,
                set_metadata={
                    'address': rec['tags'].get('address'),
                    'extensionNumber': rec['tags'].get('extensionNumber')
                }
            )
            print(f"{idx}: Updated metadata for {rec_id}.")
        # Similarly update summary vector
        summary_id = f"{rec_id}-summary"
        res2 = index.query(id=summary_id, top_k=1)
        if res2.get('matches'):
            rec = convert_record(rec)
            index.update(
                id=summary_id,
                set_metadata={
                    'address': rec['tags'].get('address'),
                    'extensionNumber': rec['tags'].get('extensionNumber')
                }
            )
            print(f"{idx}: Updated metadata for {summary_id}.")
