import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import yfinance as yf

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

#time range for query
start_date = (date.today() - relativedelta(years = 25)).strftime('%Y-%m-%d')
end_date = date.today().strftime('%Y-%m-%d')

@data_loader
def load_data_from_api(*args, **kwargs):
    
    #get the gold and silver price USD/oz from Yahoo Finance API
    yf_data = yf.download(['GC=F', 'SI=F'],
                        start = start_date,
                        end = end_date,
                        progress = False)

    #edit the columns names
    #flatten the multi-level columns
    yf_data.columns = yf_data.columns.map('_'.join).str.replace('GC=F', 'Gold').str.replace('SI=F', 'Silver')

    return yf_data

#test code to be developed
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'