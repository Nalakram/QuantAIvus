class IBConnectionError(Exception):
    pass

class DataFetchError(Exception):
    pass

class NoDataError(DataFetchError):
    def __init__(self, symbol):
        super().__init__(f"No historical data returned for {symbol}")

class DataValidationError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}

class ConfigValidationError(Exception):
    def __init__(self, message, validation_errors=None):
        super().__init__(message)
        self.validation_errors = validation_errors or []