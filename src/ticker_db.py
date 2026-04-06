"""
Ticker Database — fuzzy name-to-symbol resolver.
Covers major NYSE/NASDAQ and NSE/BSE stocks.
resolve(query) -> (symbol, name, exchange, currency)
"""

# ---------------------------------------------------------------------------
# Master ticker list  {symbol: (display_name, exchange, currency)}
# ---------------------------------------------------------------------------
TICKER_DB = {
    # ── USA — Mega / Large Cap ──────────────────────────────────────────────
    "AAPL":  ("Apple Inc",                       "NASDAQ", "USD"),
    "MSFT":  ("Microsoft Corporation",           "NASDAQ", "USD"),
    "GOOGL": ("Alphabet Inc (Google)",           "NASDAQ", "USD"),
    "GOOG":  ("Alphabet Inc Class C",            "NASDAQ", "USD"),
    "AMZN":  ("Amazon.com Inc",                  "NASDAQ", "USD"),
    "NVDA":  ("NVIDIA Corporation",              "NASDAQ", "USD"),
    "META":  ("Meta Platforms Inc",              "NASDAQ", "USD"),
    "TSLA":  ("Tesla Inc",                       "NASDAQ", "USD"),
    "AVGO":  ("Broadcom Inc",                    "NASDAQ", "USD"),
    "JPM":   ("JPMorgan Chase & Co",             "NYSE",   "USD"),
    "V":     ("Visa Inc",                        "NYSE",   "USD"),
    "MA":    ("Mastercard Inc",                  "NYSE",   "USD"),
    "XOM":   ("Exxon Mobil Corporation",         "NYSE",   "USD"),
    "UNH":   ("UnitedHealth Group",              "NYSE",   "USD"),
    "JNJ":   ("Johnson & Johnson",               "NYSE",   "USD"),
    "WMT":   ("Walmart Inc",                     "NYSE",   "USD"),
    "PG":    ("Procter & Gamble",                "NYSE",   "USD"),
    "BAC":   ("Bank of America",                 "NYSE",   "USD"),
    "HD":    ("Home Depot",                      "NYSE",   "USD"),
    "CVX":   ("Chevron Corporation",             "NYSE",   "USD"),
    "KO":    ("Coca-Cola Company",               "NYSE",   "USD"),
    "PEP":   ("PepsiCo Inc",                     "NASDAQ", "USD"),
    "ABBV":  ("AbbVie Inc",                      "NYSE",   "USD"),
    "LLY":   ("Eli Lilly and Company",           "NYSE",   "USD"),
    "AMD":   ("Advanced Micro Devices",          "NASDAQ", "USD"),
    "INTC":  ("Intel Corporation",               "NASDAQ", "USD"),
    "CSCO":  ("Cisco Systems",                   "NASDAQ", "USD"),
    "NFLX":  ("Netflix Inc",                     "NASDAQ", "USD"),
    "DIS":   ("Walt Disney Company",             "NYSE",   "USD"),
    "PYPL":  ("PayPal Holdings",                 "NASDAQ", "USD"),
    "ADBE":  ("Adobe Inc",                       "NASDAQ", "USD"),
    "CRM":   ("Salesforce Inc",                  "NYSE",   "USD"),
    "ORCL":  ("Oracle Corporation",              "NYSE",   "USD"),
    "IBM":   ("IBM Corporation",                 "NYSE",   "USD"),
    "GE":    ("GE Aerospace",                    "NYSE",   "USD"),
    "BA":    ("Boeing Company",                  "NYSE",   "USD"),
    "CAT":   ("Caterpillar Inc",                 "NYSE",   "USD"),
    "MMM":   ("3M Company",                      "NYSE",   "USD"),
    "GS":    ("Goldman Sachs Group",             "NYSE",   "USD"),
    "MS":    ("Morgan Stanley",                  "NYSE",   "USD"),
    "WFC":   ("Wells Fargo & Company",           "NYSE",   "USD"),
    "C":     ("Citigroup Inc",                   "NYSE",   "USD"),
    "BRK-B": ("Berkshire Hathaway B",            "NYSE",   "USD"),
    "UPS":   ("United Parcel Service",           "NYSE",   "USD"),
    "FDX":   ("FedEx Corporation",               "NYSE",   "USD"),
    "T":     ("AT&T Inc",                        "NYSE",   "USD"),
    "VZ":    ("Verizon Communications",          "NYSE",   "USD"),
    "AMGN":  ("Amgen Inc",                       "NASDAQ", "USD"),
    "GILD":  ("Gilead Sciences",                 "NASDAQ", "USD"),
    "SBUX":  ("Starbucks Corporation",           "NASDAQ", "USD"),
    "MCD":   ("McDonald's Corporation",          "NYSE",   "USD"),
    "NKE":   ("Nike Inc",                        "NYSE",   "USD"),
    "TGT":   ("Target Corporation",              "NYSE",   "USD"),
    "COST":  ("Costco Wholesale",                "NASDAQ", "USD"),
    "F":     ("Ford Motor Company",              "NYSE",   "USD"),
    "GM":    ("General Motors",                  "NYSE",   "USD"),
    "PFE":   ("Pfizer Inc",                      "NYSE",   "USD"),
    "MRK":   ("Merck & Co",                      "NYSE",   "USD"),
    "BMY":   ("Bristol-Myers Squibb",            "NYSE",   "USD"),
    "UBER":  ("Uber Technologies",               "NYSE",   "USD"),
    "LYFT":  ("Lyft Inc",                        "NASDAQ", "USD"),
    "SNAP":  ("Snap Inc",                        "NYSE",   "USD"),
    "SPOT":  ("Spotify Technology",              "NYSE",   "USD"),
    "COIN":  ("Coinbase Global",                 "NASDAQ", "USD"),
    "HOOD":  ("Robinhood Markets",               "NASDAQ", "USD"),
    "RBLX":  ("Roblox Corporation",              "NYSE",   "USD"),
    "PLTR":  ("Palantir Technologies",           "NYSE",   "USD"),
    "RIVN":  ("Rivian Automotive",               "NASDAQ", "USD"),
    "LCID":  ("Lucid Group",                     "NASDAQ", "USD"),
    "SOFI":  ("SoFi Technologies",               "NASDAQ", "USD"),
    "ARM":   ("Arm Holdings",                    "NASDAQ", "USD"),
    "SMCI":  ("Super Micro Computer",            "NASDAQ", "USD"),
    # ── USA — ETFs ──────────────────────────────────────────────────────────
    "SPY":   ("SPDR S&P 500 ETF",               "NYSE",   "USD"),
    "QQQ":   ("Invesco Nasdaq-100 ETF",          "NASDAQ", "USD"),
    "ITA":   ("iShares U.S. Aerospace & Defense","NYSE",   "USD"),
    "GLD":   ("SPDR Gold Shares",                "NYSE",   "USD"),
    "VTI":   ("Vanguard Total Stock Market ETF", "NYSE",   "USD"),
    "VOO":   ("Vanguard S&P 500 ETF",            "NYSE",   "USD"),
    "IVV":   ("iShares Core S&P 500 ETF",        "NYSE",   "USD"),
    "XLK":   ("Technology Select Sector SPDR",   "NYSE",   "USD"),
    "XLF":   ("Financial Select Sector SPDR",    "NYSE",   "USD"),
    "ARKK":  ("ARK Innovation ETF",              "NYSE",   "USD"),
    # ── USA — Defense ───────────────────────────────────────────────────────
    "LMT":   ("Lockheed Martin",                 "NYSE",   "USD"),
    "RTX":   ("Raytheon Technologies",           "NYSE",   "USD"),
    "NOC":   ("Northrop Grumman",                "NYSE",   "USD"),
    "GD":    ("General Dynamics",                "NYSE",   "USD"),
    "LHX":   ("L3Harris Technologies",           "NYSE",   "USD"),
    "HII":   ("Huntington Ingalls Industries",   "NYSE",   "USD"),
    # ── India — NSE ─────────────────────────────────────────────────────────
    "RELIANCE.NS":  ("Reliance Industries",      "NSE",    "INR"),
    "TCS.NS":       ("Tata Consultancy Services","NSE",    "INR"),
    "HDFCBANK.NS":  ("HDFC Bank",                "NSE",    "INR"),
    "INFY.NS":      ("Infosys Ltd",              "NSE",    "INR"),
    "ICICIBANK.NS": ("ICICI Bank",               "NSE",    "INR"),
    "HINDUNILVR.NS":("Hindustan Unilever",       "NSE",    "INR"),
    "ITC.NS":       ("ITC Limited",              "NSE",    "INR"),
    "SBIN.NS":      ("State Bank of India",      "NSE",    "INR"),
    "BHARTIARTL.NS":("Bharti Airtel",            "NSE",    "INR"),
    "BAJFINANCE.NS":("Bajaj Finance",            "NSE",    "INR"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank",      "NSE",    "INR"),
    "LT.NS":        ("Larsen & Toubro",          "NSE",    "INR"),
    "ASIANPAINT.NS":("Asian Paints",             "NSE",    "INR"),
    "AXISBANK.NS":  ("Axis Bank",                "NSE",    "INR"),
    "MARUTI.NS":    ("Maruti Suzuki",            "NSE",    "INR"),
    "SUNPHARMA.NS": ("Sun Pharmaceutical",       "NSE",    "INR"),
    "TATAMOTORS.NS":("Tata Motors",              "NSE",    "INR"),
    "WIPRO.NS":     ("Wipro Ltd",                "NSE",    "INR"),
    "ULTRACEMCO.NS":("UltraTech Cement",         "NSE",    "INR"),
    "NESTLEIND.NS": ("Nestle India",             "NSE",    "INR"),
    "POWERGRID.NS": ("Power Grid Corporation",   "NSE",    "INR"),
    "NTPC.NS":      ("NTPC Limited",             "NSE",    "INR"),
    "ONGC.NS":      ("Oil & Natural Gas Corp",   "NSE",    "INR"),
    "TATASTEEL.NS": ("Tata Steel",               "NSE",    "INR"),
    "ADANIENT.NS":  ("Adani Enterprises",        "NSE",    "INR"),
    "ADANIPORTS.NS":("Adani Ports",              "NSE",    "INR"),
    "HCLTECH.NS":   ("HCL Technologies",         "NSE",    "INR"),
    "TECHM.NS":     ("Tech Mahindra",            "NSE",    "INR"),
    "DIVISLAB.NS":  ("Divi's Laboratories",      "NSE",    "INR"),
    "DRREDDY.NS":   ("Dr. Reddy's Laboratories", "NSE",    "INR"),
    "CIPLA.NS":     ("Cipla Ltd",                "NSE",    "INR"),
    "BAJAJFINSV.NS":("Bajaj Finserv",            "NSE",    "INR"),
    "EICHERMOT.NS": ("Eicher Motors",            "NSE",    "INR"),
    "HEROMOTOCO.NS":("Hero MotoCorp",            "NSE",    "INR"),
    "M&M.NS":       ("Mahindra & Mahindra",      "NSE",    "INR"),
    "TITAN.NS":     ("Titan Company",            "NSE",    "INR"),
    "JSWSTEEL.NS":  ("JSW Steel",                "NSE",    "INR"),
    "COALINDIA.NS": ("Coal India",               "NSE",    "INR"),
    "INDUSINDBK.NS":("IndusInd Bank",            "NSE",    "INR"),
    "GRASIM.NS":    ("Grasim Industries",        "NSE",    "INR"),
    "BPCL.NS":      ("Bharat Petroleum",         "NSE",    "INR"),
    "TATACONSUM.NS":("Tata Consumer Products",   "NSE",    "INR"),
    "PIDILITIND.NS":("Pidilite Industries",      "NSE",    "INR"),
    "ZOMATO.NS":    ("Zomato Ltd",               "NSE",    "INR"),
    "PAYTM.NS":     ("One 97 Communications (Paytm)","NSE","INR"),
    "NYKAA.NS":     ("FSN E-Commerce (Nykaa)",   "NSE",    "INR"),
    "IRCTC.NS":     ("IRCTC",                    "NSE",    "INR"),
    "HAL.NS":       ("Hindustan Aeronautics",    "NSE",    "INR"),
    "BEL.NS":       ("Bharat Electronics",       "NSE",    "INR"),
    "^NSEI":        ("Nifty 50 Index",           "NSE",    "INR"),
    "^BSESN":       ("BSE Sensex Index",         "BSE",    "INR"),
}

# ---------------------------------------------------------------------------
# Alias map — common name fragments → symbol
# ---------------------------------------------------------------------------
_ALIASES = {
    # USA
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "nvidia": "NVDA", "meta": "META", "facebook": "META",
    "tesla": "TSLA", "netflix": "NFLX", "disney": "DIS", "walmart": "WMT",
    "jpmorgan": "JPM", "jp morgan": "JPM", "visa": "V", "mastercard": "MA",
    "exxon": "XOM", "chevron": "CVX", "pfizer": "PFE", "merck": "MRK",
    "johnson": "JNJ", "cocacola": "KO", "coca cola": "KO", "pepsi": "PEP",
    "mcdonalds": "MCD", "starbucks": "SBUX", "nike": "NKE", "intel": "INTC",
    "cisco": "CSCO", "oracle": "ORCL", "salesforce": "CRM", "adobe": "ADBE",
    "boeing": "BA", "caterpillar": "CAT", "goldman": "GS", "morgan stanley": "MS",
    "wells fargo": "WFC", "bank of america": "BAC", "citigroup": "C",
    "berkshire": "BRK-B", "uber": "UBER", "palantir": "PLTR", "arm": "ARM",
    "broadcom": "AVGO", "lockheed": "LMT", "raytheon": "RTX",
    "northrop": "NOC", "general dynamics": "GD", "spy": "SPY", "qqq": "QQQ",
    "nasdaq etf": "QQQ", "sp500": "SPY", "s&p": "SPY", "gold": "GLD",
    # India
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "tata consultancy": "TCS.NS",
    "hdfc bank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS", "infosys": "INFY.NS",
    "infy": "INFY.NS", "icici": "ICICIBANK.NS", "icici bank": "ICICIBANK.NS",
    "hindustan unilever": "HINDUNILVR.NS", "itc": "ITC.NS",
    "sbi": "SBIN.NS", "state bank": "SBIN.NS", "bharti": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS", "bajaj finance": "BAJFINANCE.NS",
    "kotak": "KOTAKBANK.NS", "l&t": "LT.NS", "larsen": "LT.NS",
    "asian paints": "ASIANPAINT.NS", "axis bank": "AXISBANK.NS",
    "maruti": "MARUTI.NS", "suzuki": "MARUTI.NS",
    "sun pharma": "SUNPHARMA.NS", "tata motors": "TATAMOTORS.NS",
    "wipro": "WIPRO.NS", "ultratech": "ULTRACEMCO.NS", "nestle": "NESTLEIND.NS",
    "ongc": "ONGC.NS", "tata steel": "TATASTEEL.NS", "adani": "ADANIENT.NS",
    "hcl": "HCLTECH.NS", "tech mahindra": "TECHM.NS",
    "dr reddy": "DRREDDY.NS", "cipla": "CIPLA.NS",
    "titan": "TITAN.NS", "zomato": "ZOMATO.NS", "paytm": "PAYTM.NS",
    "nykaa": "NYKAA.NS", "irctc": "IRCTC.NS", "hal": "HAL.NS",
    "hindustan aeronautics": "HAL.NS", "bharat electronics": "BEL.NS",
    "nifty": "^NSEI", "sensex": "^BSESN",
}


def resolve(query: str) -> tuple[str, str, str, str] | None:
    """
    Resolve a free-text query to (symbol, name, exchange, currency).
    Steps:
      1. Exact symbol match (case-insensitive)
      2. Alias map (e.g. "apple" → AAPL)
      3. Fuzzy substring match on name and symbol
    Returns None if no match found.
    """
    q = query.strip()
    if not q:
        return None

    qu = q.upper()

    # 1. Exact symbol match
    if qu in TICKER_DB:
        name, exch, curr = TICKER_DB[qu]
        return qu, name, exch, curr

    # 2. Alias map
    ql = q.lower()
    if ql in _ALIASES:
        sym = _ALIASES[ql]
        name, exch, curr = TICKER_DB[sym]
        return sym, name, exch, curr

    # 3. Partial alias match (e.g. "appl" hits "apple")
    for alias, sym in _ALIASES.items():
        if ql in alias or alias in ql:
            name, exch, curr = TICKER_DB[sym]
            return sym, name, exch, curr

    # 4. Symbol prefix match
    candidates = [(sym, *rest) for sym, rest in TICKER_DB.items()
                  if sym.startswith(qu) or sym.replace(".NS","").startswith(qu)]
    if candidates:
        sym, name, exch, curr = candidates[0]
        return sym, name, exch, curr

    # 5. Name substring match
    for sym, (name, exch, curr) in TICKER_DB.items():
        if ql in name.lower():
            return sym, name, exch, curr

    return None


def search(query: str, limit: int = 8) -> list[dict]:
    """
    Return up to `limit` matches for autocomplete display.
    Each result: {symbol, name, exchange, currency, label}
    """
    q  = query.strip().lower()
    qu = q.upper()
    if not q:
        return []

    scored = []
    for sym, (name, exch, curr) in TICKER_DB.items():
        score = 0
        sym_base = sym.replace(".NS", "").replace(".BO", "")
        if sym == qu or sym_base == qu:          score = 100
        elif sym.startswith(qu):                 score = 80
        elif q in _ALIASES:                      score = 90
        elif any(q in a for a in _ALIASES if _ALIASES[a] == sym): score = 85
        elif q in name.lower():                  score = 70
        elif any(w.startswith(q) for w in name.lower().split()): score = 60
        if score:
            scored.append((score, sym, name, exch, curr))

    scored.sort(key=lambda x: -x[0])
    return [
        {"symbol": s, "name": n, "exchange": e, "currency": c,
         "label": f"{s} — {n} ({e})"}
        for _, s, n, e, c in scored[:limit]
    ]


if __name__ == "__main__":
    for q in ["apple", "reliance", "tata", "NVDA", "hdfc", "boeing", "nifty", "xyz"]:
        r = resolve(q)
        print(f"  {q:20s} -> {r}")
