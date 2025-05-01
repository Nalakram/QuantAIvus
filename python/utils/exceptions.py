class IBConnectionError(Exception): 
    pass 
 
class DataFetchError(Exception): 
    pass 
 
class NoDataError(DataFetchError): 
    def __init__(self, symbol): 
        super().__init__(f"No historical data returned for {symbol}") 
