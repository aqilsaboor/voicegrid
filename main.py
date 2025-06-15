import requests
import base64
import time
import json
from datetime import datetime, timedelta
from data_transformation_pinecone import data_transformation
from update_cloud_Json_file import update_data
from credentials import client_id, client_secret


def main():
    """
    Authenticate, fetch call recordings from the last 24 hours,
    download and extract them, then transform and store data.
    """
    # --------------------
    # Step 0: Encode client credentials for OAuth Basic Auth
    # --------------------
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

    # --------------------
    # Step 1: Request OAuth access token
    # --------------------
    token_url = 'https://api.8x8.com/oauth/v2/token'
    token_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_credentials}'
    }
    token_data = {'grant_type': 'client_credentials'}
    response = requests.post(token_url, headers=token_headers, data=token_data)

    print("Token Response Status Code:", response.status_code)
    print("Token Response JSON:", response.json())
    if response.status_code != 200:
        print("Failed to get access token. Exiting.")
        exit()
    access_token = response.json().get('access_token')

    # --------------------
    # Step 2: Determine timestamp for 24 hours ago (in ms)
    # --------------------
    now = datetime.utcnow()
    since = now - timedelta(hours=24)
    since_ms = int(time.mktime(since.timetuple())) * 1000

    # --------------------
    # Step 3: Fetch recordings created in the last 24 hours
    # --------------------
    region = "us-east"
    storage_url = f'https://api.8x8.com/storage/{region}/v3/objects'
    params = {
        'filter': f'type==callrecording;createdTime=gt={since_ms}',
        'sortField': 'createdTime',
        'sortDirection': 'ASC',
        'pageKey': 0,
        'limit': 1000
    }
    storage_headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(storage_url, headers=storage_headers, params=params)
    print("Storage API Response Status Code:", response.status_code)

    # --------------------
    # Step 4: Save metadata JSON and collect recording IDs
    # --------------------
    timestamp_str = datetime.utcnow().strftime("%Y_%m_%d__%H_%M_%S")
    json_file = f"{timestamp_str}.json"
    recording_ids = []
    if response.status_code == 200:
        print("Successfully fetched recordings.")
        content = response.json().get('content', [])
        with open(json_file, 'w') as jf:
            json.dump(content, jf, indent=4)
        recording_ids = [rec.get('id') for rec in content if rec.get('id')]
    elif response.status_code == 401:
        print("Unauthorized. Check your access token.")
    else:
        print(f"Unexpected error: {response.status_code}")

    print("Total recordings fetched:", len(recording_ids))

    # --------------------
    # Step 5: Bulk download recordings if any IDs found
    # --------------------
    if recording_ids:
        start_url = f'https://api.8x8.com/storage/{region}/v3/bulk/download/start'
        dl_headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        resp = requests.post(start_url, headers=dl_headers, json=recording_ids)
        print("Download Start Status Code:", resp.status_code)
        zip_name = resp.json().get('zipName')

        # Check and retry if generation failed
        status_url = f'https://api.8x8.com/storage/{region}/v3/bulk/download/status/{zip_name}'
        status_resp = requests.get(status_url, headers=dl_headers)
        if status_resp.json().get('status') == 'FAILED':
            print("Zip generation failed; retrying...")
            resp = requests.post(start_url, headers=dl_headers, json=recording_ids)
            zip_name = resp.json().get('zipName')
            status_resp = requests.get(status_url, headers=dl_headers)
        print("Download Status:", status_resp.json())

        # Wait for ZIP to be ready, then download
        time.sleep(300)
        download_url = f'https://api.8x8.com/storage/{region}/v3/bulk/download/{zip_name}'
        final_resp = requests.get(download_url, headers=dl_headers, stream=True)
        print("Final Download Status Code:", final_resp.status_code)
        if final_resp.status_code == 200:
            with open(zip_name, 'wb') as zf:
                for chunk in final_resp.iter_content(chunk_size=8192):
                    if chunk:
                        zf.write(chunk)
            print("Download completed successfully.")
        else:
            print("Failed to download file.", final_resp.text)

        # --------------------
        # Step 6: Transform and upsert data
        # --------------------
        data_transformation(zip_name, json_file, day=f"Last day: {now}")


def run_loop():
    """
    Continuously run main() and update cloud JSON every 3 hours.
    """
    while True:
        main()
        # Update cloud-based JSON with new embeddings data
        update_data()
        print(datetime.utcnow(), ": Waiting for 3 hours for next run")
        time.sleep(3 * 3600)


if __name__ == "__main__":
    run_loop()
