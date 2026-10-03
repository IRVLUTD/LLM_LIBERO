"""Upload a fetched video selection to Google Drive and write website preview links."""
from __future__ import annotations

import argparse
import hashlib
import re
import json
from pathlib import Path

from fetch_rollout_videos import add_filters, matches, video_names, write_index

DEFAULT_FOLDER = '1D6vn9HiG9V81ROdV57m7zy8tOsDLL84M'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('outputs/website_videos'))
    parser.add_argument('--folder-id', default=DEFAULT_FOLDER)
    parser.add_argument('--credentials', type=Path, help='Downloaded Desktop OAuth client JSON; needed on first run')
    parser.add_argument('--token', type=Path, default=Path('outputs/drive_auth/token.json'))
    parser.add_argument('--public', action='store_true', help='Give anyone with the link reader access to selected videos')
    parser.add_argument('--dry-run', action='store_true')
    add_filters(parser)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.folder_id):
        parser.error('Invalid Drive folder ID')
    if args.task != 'all' and (not args.task.isdigit() or not 0 <= int(args.task) <= 9):
        parser.error('--task must be all or 0–9')
    all_rows = json.loads((args.input / 'videos.json').read_text())['videos']
    names = video_names(all_rows)
    rows = [row for row in all_rows if matches(row, args)]
    if not rows:
        parser.error('No videos match these selectors')
    root = args.input.resolve()
    for row in rows:
        path = (root / row['file']).resolve()
        if not path.is_relative_to(root) or path.suffix != '.mp4' or not path.is_file():
            parser.error(f'Invalid/missing local video: {row["file"]}')
    print(f'{len(rows)} videos selected for folder {args.folder_id}')
    if args.dry_run:
        for row in rows:
            print(names[row['key']])
        return
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        parser.error('Install google-api-python-client and google-auth-oauthlib (see docs/video_hosting.md)')
    # Existing arbitrary destination folders require full Drive scope without a Picker grant.
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_authorized_user_file(str(args.token), scopes) if args.token.exists() else None
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not args.credentials:
                parser.error('--credentials is required for first-time OAuth login')
            credentials = InstalledAppFlow.from_client_secrets_file(str(args.credentials), scopes).run_local_server(port=0, open_browser=False)
        args.token.parent.mkdir(parents=True, exist_ok=True)
        args.token.touch(mode=0o600, exist_ok=True)
        args.token.chmod(0o600)
        args.token.write_text(credentials.to_json())
    service = build('drive', 'v3', credentials=credentials)
    folder = service.files().get(fileId=args.folder_id, fields='mimeType,capabilities(canAddChildren)', supportsAllDrives=True).execute()
    if folder['mimeType'] != 'application/vnd.google-apps.folder' or not folder.get('capabilities', {}).get('canAddChildren'):
        parser.error('Destination must be a folder you can upload into')
    output = args.input / 'drive_videos.json'
    completed = {row['key']: row for row in json.loads(output.read_text())['videos'] if row.get('drive_folder_id') == args.folder_id} if output.exists() else {}
    for row in rows:
        key = row['key']
        name = names[key]
        query = f"'{args.folder_id}' in parents and trashed = false and appProperties has {{ key='rolloutKey' and value='{key}' }}"
        found = service.files().list(q=query, fields='files(id,name,md5Checksum)', supportsAllDrives=True, includeItemsFromAllDrives=True).execute()['files']
        media = MediaFileUpload(str(root / row['file']), mimetype='video/mp4', resumable=True)
        with (root / row['file']).open('rb') as stream:
            digest = hashlib.md5()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if len(found) > 1:
            parser.error(f'Multiple Drive files match {key}; resolve duplicates before continuing')
        if found and found[0].get('md5Checksum') == digest.hexdigest():
            file_id = found[0]['id']
            if found[0].get('name') != name:
                service.files().update(fileId=file_id, body={'name': name}, fields='id', supportsAllDrives=True).execute()
        else:
            if found:
                request = service.files().update(fileId=found[0]['id'], body={'name': name}, media_body=media, fields='id', supportsAllDrives=True)
            else:
                request = service.files().create(body={'name': name, 'parents': [args.folder_id], 'appProperties': {'rolloutKey': key}}, media_body=media, fields='id', supportsAllDrives=True)
            response = None
            while response is None:
                _, response = request.next_chunk(num_retries=5)
            file_id = response['id']
        if args.public:
            service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}, supportsAllDrives=True).execute()
        completed[key] = dict(row, drive_name=name, drive_folder_id=args.folder_id, drive_file_id=file_id, preview_url=f'https://drive.google.com/file/d/{file_id}/preview', view_url=f'https://drive.google.com/file/d/{file_id}/view')
        write_index(output, list(completed.values()))
        print(f'{row["model"]} {row["reasoning"]} task={row["task_id"]} state={row["init_state"]}: {file_id}', flush=True)
    print(f'Website index: {output}')


if __name__ == '__main__':
    main()
