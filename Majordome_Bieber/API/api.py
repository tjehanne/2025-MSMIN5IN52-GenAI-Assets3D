from flask import Flask, jsonify
from flask_cors import CORS
from Gmail_API import readMail

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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
