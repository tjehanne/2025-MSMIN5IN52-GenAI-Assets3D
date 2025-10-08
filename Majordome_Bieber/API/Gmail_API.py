import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from email.mime.text import MIMEText

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify",
          "https://www.googleapis.com/auth/gmail.readonly"]


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


def createDraft():
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


def readMail(nbr_mail=10):
    """Affiche l'utilisation de base de l'API Gmail.
    Récupère et affiche les 10 derniers e-mails de l'utilisateur.
    """
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # S'il n'y a pas d'informations d'identification (valides) disponibles, laissez l'utilisateur se connecter.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "Majordome_Bieber/API/credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Enregistrez les informations d'identification pour la prochaine exécution
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        # Appeler l'API Gmail
        service = build("gmail", "v1", credentials=creds)

        # Récupérer la liste des 10 derniers messages.
        results = service.users().messages().list(
            userId="me", maxResults=nbr_mail).execute()
        messages = results.get("messages", [])

        if not messages:
            return []

        emails_list = []
        for message in messages:
            msg = service.users().messages().get(
                userId="me", id=message["id"]).execute()
            payload = msg.get("payload", {})
            headers = payload.get("headers", [])

            subject = ""
            sender = ""
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                if header['name'] == 'From':
                    sender = header['value']

            snippet = msg.get("snippet", "")
            emails_list.append({
                "from": sender,
                "subject": subject,
                "snippet": snippet,
                "id": message["id"]
            })

        return emails_list

    except HttpError as error:
        print(f"An error occurred: {error}")
