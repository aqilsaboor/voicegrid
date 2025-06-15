from pinecone import Pinecone
import json
from google.cloud import storage
from google.oauth2 import service_account
from credentials import pinecone_key

# Initialize Pinecone client with API key
pc = Pinecone(api_key=pinecone_key)

# Function to sync new embeddings from Pinecone index to GCS JSON
def update_data():
    # Connect to the existing Pinecone index
    index = pc.Index("8x8-transcription-embeddings")

    # Gather all vector IDs present in the index
    ids = []
    for item in index.list():
        ids.extend(item)

    # Exclude summary entries to avoid duplicate records
    unique_ids = [id for id in ids if not id.endswith("-summary")]
    print(f"Found {len(unique_ids)} unique record IDs to verify")

    # Authenticate to Google Cloud Storage using service account key file
    credentials = service_account.Credentials.from_service_account_file("c-storage.json")
    client = storage.Client(credentials=credentials, project=credentials.project_id)

    # Reference the bucket and blob containing existing chat bot data
    bucket = client.bucket("Calls-transcriptions")
    blob = bucket.blob("Chat_bot_Data.json")

    # Download current JSON data from GCS
    json_str = blob.download_as_text()
    cloud_data = json.loads(json_str)
    cloud_data_ids = [r["ID"] for r in cloud_data]

    # Iterate over each unique vector ID and append new entries if missing
    for count, rec_id in enumerate(unique_ids, start=1):
        if rec_id not in cloud_data_ids:
            print(f"{count}/{len(unique_ids)}: Adding new ID {rec_id}")

            # Query Pinecone for metadata without values to reduce payload
            result = index.query(
                id=rec_id,
                top_k=1,
                include_metadata=True,
                include_values=False
            )

            # Skip if no match found
            if not result.matches:
                print(f"No metadata found for ID {rec_id}")
                continue

            metadata = result.matches[0].metadata

            # Ensure transcription field exists
            if 'transcription' not in metadata:
                print(f"Skipping {rec_id}: missing transcription")
                continue

            # Normalize startTime to seconds if needed
            start_time = metadata.get('startTime', 'N/A')
            try:
                # If timestamp in ms, convert to seconds
                if isinstance(start_time, (int, float)) and len(str(int(start_time))) > 10:
                    start_time = start_time / 1000
            except Exception:
                pass

            # Build new record payload
            new_record = {
                "ID": rec_id,
                "transcription": metadata.get('transcription', 'N/A'),
                "startTime": start_time,
                "patient_name": metadata.get('patient_name', 'N/A'),
                "doctor_name": metadata.get('doctor_name', 'N/A'),
            }
            # Add to local list
            cloud_data.append(new_record)

    # Write updated data back to local file
    with open('Chat_bot_Data.json', 'w') as f:
        json.dump(cloud_data, f, indent=4)

    # Upload updated JSON content to GCS
    updated_json = json.dumps(cloud_data, indent=2)
    blob.upload_from_string(updated_json, content_type='application/json')

    print("\nAll new records synced to GCS successfully!")

# Entry point
if __name__ == "__main__":
    update_data()