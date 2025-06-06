# Utility per il parsing JSON
import re
import json


class JSONParser:
    """Utility per il parsing di JSON da testi con virgolette"""
    
    @staticmethod
    def extract_json(text):
        """Estrae JSON da stringhe con virgolette"""
        try:
            # Rimuovi virgolette esterne se presenti
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            # Trova il primo oggetto JSON valido
            match = re.search(r"\{[\s\S]*?\}", text)
            if match:
                json_string = match.group(0)
                return json.loads(json_string)
            else:
                return None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def extract_score_from_error(error_message):
        """Estrae lo score da un messaggio di errore"""
        # Cerca pattern come "score: 0.8" o "value=0.5"
        match = re.search(r"(score|value)[:=]\s*([0-9.]+)", error_message, re.IGNORECASE)
        if match:
            try:
                return float(match.group(2))
            except ValueError:
                return None
        return None