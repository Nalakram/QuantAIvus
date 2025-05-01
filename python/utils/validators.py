def validate_symbol(symbol): 
    if not isinstance(symbol, str) or not symbol.isalnum(): 
        raise ValueError("Invalid symbol") 
def validate_date(date): 
    from datetime import datetime 
    try: 
        if date and not datetime.strptime(date, "%%Y%%m%%d %%H:%%M:%%S"): 
            raise ValueError("Invalid date format") 
    except ValueError: 
        raise ValueError("Invalid date format") 
