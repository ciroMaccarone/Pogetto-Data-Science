from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List, Tuple
from dateutil.relativedelta import relativedelta
from datetime import datetime
import yfinance as yf 
import requests 

class ActionGetStockPrice(Action):
    def name(self) -> Text:
        return "action_get_stock_price"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Recupera il nome dell'azienda dallo slot "company"
            company_name = tracker.get_slot("company")

            if not company_name:
                dispatcher.utter_message(text="Mi dispiace, ma il nome dell'azienda manca.")
                return []

            # Utilizza l'azione ActionGetCompanySymbol per ottenere il simbolo dell'azienda
            symbol = self.get_company_symbol(company_name)

            if not symbol:
                dispatcher.utter_message(text=f"Mi dispiace, ma non ho potuto trovare il simbolo per {company_name}.")
                return []

            # Utilizza il simbolo dell'azienda per ottenere il prezzo delle azioni
            stock_price = self.get_stock_price(symbol)

            if not stock_price:
                dispatcher.utter_message(text=f"Mi dispiace, ma non ho potuto trovare il prezzo delle azioni per {company_name}.")
                return []

            dispatcher.utter_message(text=f"Ultimo prezzo di chiusura per {company_name} ({symbol}) : {stock_price} $")

        except Exception as e:
            dispatcher.utter_message(text=f"Si è verificato un errore durante l'ottenimento del prezzo delle azioni: {str(e)}")

        return []

    def get_company_symbol(self, company_name: str) -> str:
        try:
            api_key = "LLYI803FYT69EO96"
            base_url = "https://www.alphavantage.co/query"

            # Effettua una richiesta per ottenere il simbolo dell'azienda
            response = requests.get(base_url, params={
                "function": "SYMBOL_SEARCH",
                "keywords": company_name,
                "apikey": api_key
            })

            data = response.json()

            if "bestMatches" in data and data["bestMatches"]:
                # Estrai il simbolo dalla prima corrispondenza (se disponibile)
                symbol = data["bestMatches"][0]["1. symbol"]
                return symbol
            else:
                return None
        except Exception as e:
            print(f"Errore durante la ricerca del simbolo: {str(e)}")
            return None

    def get_stock_price(self, symbol: str) -> str:
        try:
            api_key = "YOUR_ALPHA_VANTAGE_API_KEY"
            base_url = "https://www.alphavantage.co/query"

            # Effettua una richiesta per ottenere il prezzo delle azioni
            response = requests.get(base_url, params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": api_key
            })

            data = response.json()

            if "Time Series (Daily)" in data:
                # Estrai il prezzo di chiusura dell'ultima giornata
                last_date = list(data["Time Series (Daily)"].keys())[0]
                stock_price = data["Time Series (Daily)"][last_date]["4. close"]
                return stock_price
            else:
                return None
        except Exception as e:
            print(f"Errore durante l'ottenimento del prezzo delle azioni: {str(e)}")
            return None

class action_search_news(Action):

    def name(self) -> Text:
        return "action_search_news"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        company = tracker.get_slot("company")
        # In questo esempio utilizziamo l'API di NewsAPI per cercare le notizie più recenti sull'azienda specificata dall'utente
        url = f"https://newsapi.org/v2/everything?q={company} stock&sortBy=publishedAt&apiKey=8def772c8a63483895bc65b73b1c7a85&language=en"

        response = requests.get(url)
        articles = response.json()["articles"]
        
        # Costruiamo una lista di titoli e URL dei primi 5 articoli trovati
        titles_and_urls = [(article["title"], article["url"]) for article in articles[:5]]
        
        if len(titles_and_urls) > 0:
            dispatcher.utter_message(f"Ecco le ultime notizie su {company}:")
            for title, url in titles_and_urls:
                dispatcher.utter_message(title)
                dispatcher.utter_message(url)
        else:
            dispatcher.utter_message(f"Mi dispiace, non ho trovato notizie su {company}.")

        return []