import datetime
import os.path
from plyer import notification
import time
import threading

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

def list_events(service, notify=False):
    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
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
    result = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        summary = event.get("summary", "Sans titre")
        location = event.get("location", "")
        result.append({"summary": summary, "start": start, "location": location})
        if notify:
            notification.notify(
                title="Événement à venir",
                message=f"{summary}\nDébut : {start}",
                timeout=10
            )
    return result

def create_event(service, summary, date, start_time, end_time, location=""):
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
        'location': location
    }
    created_event = service.events().insert(calendarId='primary', body=event).execute()
    notification.notify(
        title="Événement créé",
        message=f"{summary}\nDébut : {start}",
        timeout=10
    )
    return created_event.get('id')

def notify_events_at_time(service):
    """
    Envoie une notification exactement à l'heure de début de chaque événement du jour.
    """
    notified_events = set()
    while True:
        now = datetime.datetime.now(datetime.timezone.utc)
        # Cherche les événements du jour
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + datetime.timedelta(days=1)
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        for event in events:
            event_id = event["id"]
            start_str = event["start"].get("dateTime", event["start"].get("date"))
            try:
                event_time = datetime.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            except Exception:
                continue
            # Si l'événement commence maintenant (à la minute près) et pas déjà notifié
            if (now.replace(second=0, microsecond=0) == event_time.replace(second=0, microsecond=0)
                and event_id not in notified_events):
                summary = event.get("summary", "Sans titre")
                notification.notify(
                    title="C'est l'heure de l'événement !",
                    message=f"{summary}\nDébute maintenant.",
                    timeout=10
                )
                notified_events.add(event_id)
        time.sleep(30)  # Vérifie toutes les 30 secondes

def main():
    creds = get_credentials()
    try:
        service = build("calendar", "v3", credentials=creds)
        # Lancer les notifications en arrière-plan
        notif_thread = threading.Thread(target=notify_events_at_time, args=(service,), daemon=True)
        notif_thread.start()
        print("Notifications à l'heure des événements activées en arrière-plan.")
        print("1. Lire les événements")
        print("2. Créer un événement")
        while True:
            choix = input("Choix (1/2, Ctrl+C pour quitter) : ")
            if choix == "1":
                list_events(service)
            elif choix == "2":
                summary = input("Titre de l'événement : ")
                date = input("Date (YYYY-MM-DD) : ")
                start_time = input("Heure de début (HH:MM, 24h) : ")
                end_time = input("Heure de fin (HH:MM, 24h) : ")
                location = input("Lieu (optionnel) : ")
                create_event(service, summary, date, start_time, end_time, location)
            else:
                print("Choix invalide.")
    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()