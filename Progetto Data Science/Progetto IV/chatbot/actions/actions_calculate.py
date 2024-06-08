from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List, Tuple
from dateutil.relativedelta import relativedelta
from datetime import datetime
import yfinance as yf 
import requests
import re

class ActionCalculateReturns(Action):
    def name(self) -> Text:
        return "action_calculate_returns"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            period = tracker.latest_message.get('text', '')
            start_date, end_date = parse_period(period)
            if not start_date or not end_date:
                dispatcher.utter_message(text="Non ho compreso il periodo specificato.")
                dispatcher.utter_message('Data di riferimento :', start_date)
            else:
                dispatcher.utter_message(f'Data di riferimento : {start_date}')
            company_name = tracker.get_slot("company")

            if not company_name:
                dispatcher.utter_message(text="Mi dispiace, ma il nome dell'azienda manca.")
                return []
            symbol = get_company_symbol(company_name)
            if "Information" in symbol:
                raise Exception(dispatcher.utter_message(text=symbol["Information"]))

            if not symbol:
                dispatcher.utter_message(text=f"Mi dispiace, ma non ho potuto trovare il simbolo per {company_name}.")
                return []
            data = yf.download(symbol, start=start_date, end=end_date)

            if data.empty:
                dispatcher.utter_message(text=f"Mi dispiace, ma non ho potuto trovare dati per {company_name} nel periodo specificato.")
            else:
                if "Close" in data:
                    returns = ((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100
                    dispatcher.utter_message(text=f"I rendimenti di {company_name} ({symbol}) nel periodo specificato sono del {round(returns, 2)}%")
                else:
                    dispatcher.utter_message(text=f"I dati scaricati non contengono prezzi di chiusura validi.")

        except Exception as e:
            dispatcher.utter_message(text=f"Si è verificato un errore durante il calcolo dei rendimenti: {str(e)}")

class ActionCalculateVolume(Action):
    def name(self) -> Text:
        return "action_calculate_volume"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            period = tracker.latest_message.get('text', '')
            start_date, end_date = parse_period(period)
            if not start_date or not end_date:
                dispatcher.utter_message(text="Non ho compreso il periodo specificato.")
                dispatcher.utter_message('Data di riferimento :', start_date)
                return []
            else:
                dispatcher.utter_message(f'Data di riferimento : {start_date}')
            company_name = tracker.get_slot("company")

            if not company_name:
                dispatcher.utter_message(text="Mi dispiace, ma il nome dell'azienda manca.")
                return []
            company_symbol = get_company_symbol(company_name)
            if "Information" in company_symbol:
                raise Exception(dispatcher.utter_message(text=company_symbol["Information"]))


            if not company_symbol:
                dispatcher.utter_message(text=f"Mi dispiace, ma non ho potuto trovare il simbolo per {company_name}.")
                return []
            data = yf.download(company_symbol, start=start_date, end=end_date)

            if data.empty:
                dispatcher.utter_message(text=f"Mi dispiace, ma non ho potuto trovare dati per {company_name} nel periodo specificato.")
            else:
                average_volume = data["Volume"].mean()
                dispatcher.utter_message(text=f"Il volume di scambi per {company_name} ({company_symbol}) nel periodo specificato è {round(average_volume, 2)} $")

        except Exception as e:
            dispatcher.utter_message(text=f"Si è verificato un errore durante il calcolo del volume di scambi: {str(e)}")

def get_company_symbol(company_name: str) -> str:
        try:
            api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
            base_url = "https://www.alphavantage.co/query"
            response = requests.get(base_url, params={
                "function": "SYMBOL_SEARCH",
                "keywords": company_name,
                "apikey": api_key
            })
            
            data = response.json()
            if "bestMatches" in data and data["bestMatches"]:
                symbol = data["bestMatches"][0]["1. symbol"]
                return symbol
            elif "limit" in data['Information']:
                return data
            else:
                return None
        except Exception as e:
            print(f"Errore durante la ricerca del simbolo: {str(e)}")
            return None

def parse_period(period_text: str) -> Tuple[datetime, datetime]:
        start_date = None
        end_date = datetime.now()
        try:
            if "1 mese" in period_text:
                start_date = end_date - relativedelta(months=1)
            elif "3 mesi" in period_text:
                start_date = end_date - relativedelta(months=3)
            elif "6 mesi" in period_text:
                start_date = end_date - relativedelta(months=6)
            elif "1 anno" in period_text:
                start_date = end_date - relativedelta(years=1)
            elif "2 anni" in period_text:
                start_date = end_date - relativedelta(years=2)
            else:
                start_date = end_date - relativedelta(**parse_date(period_text))
        except Exception as e:
            print("Period expression not recognized")

        return start_date, end_date

def parse_date(text):
    pattern = r'(\d+)\s+(mese|mese|m|anno|anni|a)'
    matches = re.findall(pattern, text.lower())
    parsed_date = {}
    for num, unit in matches:
        num = int(num)
        if "mese" in unit or "m" in unit:
            parsed_date["months"] = num
        elif "anno" in unit or "anni" in unit or "a" in unit:
            parsed_date["years"] = num
    
    return parsed_date