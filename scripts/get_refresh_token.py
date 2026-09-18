"""
RUN THIS ONCE, ON YOUR OWN COMPUTER (not in the cloud), to generate the
YouTube refresh token needed for unattended uploads.

Steps:
1. pip install google-auth-oauthlib google-api-python-client
2. Download your OAuth "Desktop app" client_secret.json from Google Cloud
   Console (APIs & Services -> Credentials).
3. Place it next to this script as client_secret.json.
4. Run: python get_refresh_token.py
5. A browser window opens -> log in with the Google account that owns
   your YouTube channel -> grant permission.
6. The script prints a refresh_token. Copy it into your GitHub secret
   YT_REFRESH_TOKEN (and client_id / client_secret into YT_CLIENT_ID /
   YT_CLIENT_SECRET). You never need to run this again unless you revoke
   access.
"""
import os
import sys
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

candidates = [
    "client_secret.json",
    os.path.join(os.path.dirname(__file__), "client_secret.json"),
    os.path.join(os.path.dirname(__file__), "..", "client_secret.json"),
]

client_secret_path = next((p for p in candidates if os.path.isfile(p)), None)

if not client_secret_path:
    print("\n[ERROR] 'client_secret.json' not found.")
    print("Please download your OAuth Desktop Client secret file from Google Cloud Console,")
    print("rename it to 'client_secret.json', and place it in the project directory.")
    sys.exit(1)

flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
creds = flow.run_local_server(port=0)

print("\n--- SAVE THESE AS GITHUB SECRETS ---")
print("YT_CLIENT_ID     =", creds.client_id)
print("YT_CLIENT_SECRET =", creds.client_secret)
print("YT_REFRESH_TOKEN =", creds.refresh_token)
