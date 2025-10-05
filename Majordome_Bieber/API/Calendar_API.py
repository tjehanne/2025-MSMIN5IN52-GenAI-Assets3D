import datetime
import os.path
from plyer import notification

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_credentials():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            credentials_path = os.path.join(base_dir, "credentials.json")
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

def list_events(service):
    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    print("Getting the upcoming 10 events")
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=10,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
        print("No upcoming events found.")
        return

    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        print(start, event["summary"])
        notification.notify(
            title="Événement à venir",
            message=f"{event['summary']}\nDébut : {start}",
            timeout=10
        )

def create_event(service):
    summary = input("Titre de l'événement : ")
    date = input("Date (YYYY-MM-DD) : ")
    start_time = input("Heure de début (HH:MM, 24h) : ")
    end_time = input("Heure de fin (HH:MM, 24h) : ")

    start = f"{date}T{start_time}:00"
    end = f"{date}T{end_time}:00"

    event = {
        'summary': summary,
        'start': {
            'dateTime': start,
            'timeZone': 'Europe/Paris',
        },
        'end': {
            'dateTime': end,
            'timeZone': 'Europe/Paris',
        },
    }

    event = service.events().insert(calendarId='primary', body=event).execute()
    print(f"Événement créé : {event.get('htmlLink')}")
    notification.notify(
        title="Événement créé",
        message=f"{summary}\nDébut : {start}",
        timeout=10
    )

def main():
    creds = get_credentials()
    try:
        service = build("calendar", "v3", credentials=creds)
        print("1. Lire les événements")
        print("2. Créer un événement")
        choix = input("Choix (1/2) : ")
        if choix == "1":
            list_events(service)
        elif choix == "2":
            create_event(service)
        else:
            print("Choix invalide.")
    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()