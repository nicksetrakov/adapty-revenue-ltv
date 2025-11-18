import os

from dotenv import load_dotenv

load_dotenv()
APP_API_KEYS = {
    "AFC Live – for Arsenal fans": os.getenv("ARSENAL_API_KEY"),
    "Barcelona Live – Soccer app": os.getenv("BARCELONA_API_KEY"),
    "Bayern Live – Fussball App": os.getenv("BAYERN_API_KEY"),
    "Bianconeri Live: Аpp di calcio": os.getenv("JUVENTUS_API_KEY"),
    "Blues Live: soccer app": os.getenv("CHELSEA_API_KEY"),
    "Dortmund Live - Inoffizielle": os.getenv("DORTMUND_API_KEY"),
    "Inter Live: Risultati  notizie": os.getenv("INTER_API_KEY"),
    "LFC Live: for Liverpool fans": os.getenv("LIVERPOOL_API_KEY"),
    "Real Live – soccer app": os.getenv("REAL_API_KEY"),
    "Manchester Live – United fans": os.getenv("MANCHESTER_API_KEY"),
    "Paris Foot Direct: no officiel": os.getenv("PSG_API_KEY"),
    "Rossoneri Live: no ufficiale": os.getenv("MILAN_API_KEY"),
    "Tribuna.com UA: Спорт України": os.getenv("TRIBUNA_UA_API_KEY"),
    "Футбол України: Tribuna.com UA": os.getenv("FOOTBALL_UA_API_KEY"),
    "Football Xtra - Tribuna.com": os.getenv("Soccer_Xtra_API_KEY")
    # Добавь остальные приложения сюда
}

