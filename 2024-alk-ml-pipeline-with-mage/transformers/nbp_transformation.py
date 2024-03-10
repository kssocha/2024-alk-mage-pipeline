import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, data_2, *args, **kwargs):
    
    # Specify your transformation logic here

    #nbp_au_data
    #convert json data format to DataFrame
    if data:
        data = pd.DataFrame(data)
            
    #edit the columns names
    new_columns_names = {
        'data' : 'Date',
        'cena' : 'NBP Gold Price'
    }

    data = data.rename (columns = new_columns_names)

    #convert 'Date' column to index
    data = data.set_index('Date')
    #convert gram price to ounce price
    #Banker's rounding used (round half to even)
    oz_g = 31.1034768
    data['NBP Gold Price PLN_oz'] = round(data['NBP Gold Price'] * oz_g, 4)

    #nbp_exchange_rate_data
    #convert json data format to DataFrame
    if data_2:
        data_2 = pd.DataFrame(data_2['rates'])

    currency_code = 'USD'
            
    #edit the columns names
    new_columns_names = {
        'no' : 'Table Number',
        'effectiveDate' : 'Date',
        'mid' : 'Average Exchange Rate of ' + str(currency_code) #applicable only for A Table
        #'bid' : 'Buying Rate', #applicable only for C Table
        #'ask' : 'Selling Rate' #applicable only for C Table
    }

    data_2 = data_2.rename (columns = new_columns_names)

    #convert 'Date' column to index
    data_2 = data_2.set_index('Date')

    #joining dataframes
    nbp_df = data.join(data_2)

    return nbp_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'