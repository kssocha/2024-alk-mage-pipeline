from datetime import date
from dateutil.relativedelta import relativedelta
import requests

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

#time range for all queries
start_date = (date.today() - relativedelta(years = 1)).strftime('%Y-%m-%d')
end_date = date.today().strftime('%Y-%m-%d')

@data_loader
def load_data_from_api(*args, **kwargs):
    
    url = f"https://api.nbp.pl/api/cenyzlota/{start_date}/{end_date}"
    response = requests.get(url)

    if response.status_code == 200:
        nbp_au_data = response.json()
        return nbp_au_data
    elif response.status_code == 404:
        print(f'Error: Failed to fetch data. Status code: {response.status_code} Not Found')
        return None
    elif response.status_code == 400:
        print(f'Error: Failed to fetch data. Status code: {response.status_code} Bad Request')
        return None

#test code to be developed
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'