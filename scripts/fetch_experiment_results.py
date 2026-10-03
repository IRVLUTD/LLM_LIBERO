"""Download the private results JSON from Google Drive into another website repo."""
import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file-id', required=True)
    parser.add_argument('--token', type=Path, required=True, help='OAuth authorized-user token with access; keep outside public assets')
    parser.add_argument('--output', type=Path, default=Path('public/data/experiment_results.json'))
    args = parser.parse_args()
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    credentials = Credentials.from_authorized_user_file(str(args.token))
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    service = build('drive', 'v3', credentials=credentials)
    request = service.files().get_media(fileId=args.file_id, supportsAllDrives=True)
    stream = io.BytesIO()
    downloader = MediaIoBaseDownload(stream, request)
    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=5)
    payload = stream.getvalue()
    data = json.loads(payload)
    if data.get('version') != 1 or not isinstance(data.get('episodes'), list):
        raise ValueError('Drive file is not a supported experiment results export')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_bytes(payload)
    temporary.replace(args.output)
    print(f'Downloaded {len(data["episodes"])} episodes to {args.output}')


if __name__ == '__main__':
    main()
