def validate_symbol(symbol):
    if not isinstance(symbol, str) or not symbol.isalnum():
        raise ValueError("Invalid symbol")

def validate_date(date):
    from datetime import datetime
    if not date:  # Allow empty date (fetch_historical_data permits this)
        return
    try:
        datetime.strptime(date, "%Y%m%d %H:%M:%S")
    except ValueError:
        raise ValueError("Invalid date format")