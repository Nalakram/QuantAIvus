config = {
  'ib_api': {
    'host': '127.0.0.1',
    'port': 7497,
    'client_id': 1,
    'what_to_show': 'TRADES',
    'use_rth': True,
    'format_date': 1
  },
  'data_source': {
    'type': 'CSV',
    'path': 'data/raw/historical_prices_ib.csv',
    'chunksize': 100000,
    'dask': False
  },
  'cleaning': {
    'missing_values': {
      'method': 'forward_fill',
      'params': {
        'backward_fill': True
      }
    },
    'outliers': {
      'method': 'zscore',
      'params': {
        'threshold': 3.0
      }
    },
    'denoising': {
      'method': 'ewm',
      'params': {
        'span': 10
      }
    }
  }
}
