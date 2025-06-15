import requests
import base64
from datetime import datetime, timedelta
import time
import json
from data_transformation_pinecone import data_transformation
from credentials import client_id, client_secret

# Loop over the past 300 days to fetch and download call recordings for each day interval
for day in range(1, 300):
    print("\n\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Day: {day}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

    # Get the current UTC time for timestamping
    current_time = datetime.utcnow()
    
    # Calculate the time window for this iteration:
    # 'from_datetime' is exactly 'day' days ago, and 'till_datetime' is one day more recent
    till_datetime = current_time - timedelta(days=day - 1)
    from_datetime = current_time - timedelta(days=day)
    
    # Convert datetimes to ISO 8601 format with Zulu timezone suffix
    from_date = from_datetime.isoformat() + "Z"
    till_date = till_datetime.isoformat() + "Z"

    print(f"Day {day}: From Date: {from_date}  Till Date: {till_date}")
    
    # Create a human-readable description for logging and downstream processing
    day_description = f"Last {day} day: {from_datetime} till {till_datetime}"
        
    # --------------------
    # Step 0: Encode client credentials for OAuth Basic Auth
    # --------------------
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    # --------------------
    # Step 1: Get an OAuth access token using client credentials grant
    # --------------------
    token_url = 'https://api.8x8.com/oauth/v2/token'
    token_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_credentials}'
    }
    token_data = {
        'grant_type': 'client_credentials'
    }

    response = requests.post(token_url, headers=token_headers, data=token_data)
    print("Token Response Status Code:", response.status_code)
    print("Token Response JSON:", response.json())

    # Exit if authentication fails
    if response.status_code != 200:
        print("Failed to get access token. Exiting.")
        exit()

    access_token = response.json()['access_token']

    # --------------------
    # Step 2: Fetch list of call recordings created in the date range
    # --------------------
    region = "us-east"  # Replace if using another region
    storage_url = f'https://api.8x8.com/storage/{region}/v3/objects'
    storage_params = {
        'filter': f'type==callrecording;createdTime=gt={from_date};createdTime=lt={till_date}',
        'sortField': 'createdTime',
        'sortDirection': 'ASC',
        'pageKey': 0,
        'limit': 1000
    }
    storage_headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.get(storage_url, headers=storage_headers, params=storage_params)
    print("Storage API Response Status Code:", response.status_code)

    # Generate a timestamped filename for the JSON metadata
    timestamp = str(datetime.utcnow())
    json_file_name = timestamp[:timestamp.find(".")].replace(" ","__").replace(":","_") + ".json"

    recording_ids = []  # List to collect recording IDs

    if response.status_code == 401:
        # Unauthorized: token might be invalid or expired
        print("Unauthorized. Check your access token or permissions.")
    elif response.status_code == 200:
        print("Successfully fetched recordings.")
        recordings = response.json().get('content', [])
        # Save the recordings metadata to a JSON file
        with open(json_file_name, "w") as json_file:
            json.dump(recordings, json_file, indent=4)

        # Extract recording IDs for bulk download
        for recording in recordings:
            recording_id = recording.get('id')
            if recording_id:
                recording_ids.append(recording_id)
    else:
        print(f"Unexpected error: {response.status_code}")

    print("\nTotal Recordings: ", len(recording_ids))

    # --------------------
    # Step 3: Initiate bulk download of recordings if any are found
    # --------------------
    if recording_ids:
        download_start_url = f'https://api.8x8.com/storage/{region}/v3/bulk/download/start'
        download_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        # POST the list of IDs to begin creating the ZIP archive
        response = requests.post(download_start_url, headers=download_headers, json=recording_ids)
        print("\nDownload Start Status Code:", response.status_code)
        print("Response JSON:", response.json())

        # Get the name of the created ZIP file
        zip_name = response.json().get("zipName")

        # Check the status of the ZIP preparation
        status_url = f'https://api.8x8.com/storage/{region}/v3/bulk/download/status/{zip_name}'
        response = requests.get(status_url, headers=download_headers)
        print("\nDownload Status Status Code:", response.status_code)
        print("Response JSON:", response.json())

        # Wait briefly before attempting to download the ZIP
        time.sleep(300)

        # --------------------
        # Step 4: Download the ZIP archive
        # --------------------
        download_url = f'https://api.8x8.com/storage/{region}/v3/bulk/download/{zip_name}'
        response = requests.get(download_url, headers=download_headers, stream=True)
        print("\nDownload Status Code:", response.status_code)

        # Retry once on server error
        if response.status_code == 500:
            time.sleep(300)
            response = requests.get(download_url, headers=download_headers, stream=True)
            print("\nDownload Status Code (retry):", response.status_code)
            
        # Save the ZIP file locally if successful
        if response.status_code == 200:
            with open(zip_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download completed successfully!")
        else:
            print(f"Failed to download. Status Code: {response.status_code}")
            print(response.text)

        # --------------------
        # Step 5: Transform data with the custom pipeline
        # --------------------
        data_transformation(zip_name, json_file_name, day=day_description)
