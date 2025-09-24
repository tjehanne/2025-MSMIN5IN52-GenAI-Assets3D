import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def create_draft(service):
    """Create and insert a draft email.
    """
    message = MIMEText("hello world")
    message['to'] = ""  # laissé vide pour le brouillon
    message['subject'] = "Test Draft"

    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    create_message = {
        'message': {
            'raw': encoded_message
        }
    }

    try:
        draft = service.users().drafts().create(
            userId="me", body=create_message).execute()
        print(F'Draft id: {draft["id"]}\nDraft created successfully.')
        return draft
    except HttpError as error:
        print(F'An error occurred: {error}')
        return None


def main():
    """Shows basic usage of the Gmail API.
    Creates a draft email.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "Majordome_Bieber/API/credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build("gmail", "v1", credentials=creds)
        create_draft(service)

    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()
