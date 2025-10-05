from flask import Flask, jsonify, request
from flask_cors import CORS
from Gmail_API import readMail
from Calendar_API import get_credentials, list_events, create_event
from googleapiclient.discovery import build
import datetime

app = Flask(__name__)
CORS(app)


@app.route('/api/emails')
def get_emails():
    try:
        emails = []
        # Capturer la sortie de readMail dans une structure de données
        # au lieu de l'imprimer
        creds = None
        results = readMail(10)  # Récupérer les 10 derniers emails
        return jsonify({"emails": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/calendar/events')
def get_calendar_events():
    try:
        creds = get_credentials()
        service = build("calendar", "v3", credentials=creds)
        events = list_events(service)  # Utilise ta fonction
        return jsonify({"events": events})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/calendar/events', methods=['POST'])
def create_calendar_event():
    try:
        data = request.get_json()
        summary = data.get("summary")
        date = data.get("date")
        start_time = data.get("start_time")
        end_time = data.get("end_time")
        location = data.get("location", "")
        creds = get_credentials()
        service = build("calendar", "v3", credentials=creds)
        event_id = create_event(service, summary, date, start_time, end_time, location)
        return jsonify({"success": True, "event_id": event_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
