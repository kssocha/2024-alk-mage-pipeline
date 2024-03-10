import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    
    data.index = pd.to_datetime(data.index, format = '%Y-%m-%dT00:00:00').strftime('%Y/%m/%d')
    data = data.dropna(how='any')
    data = data[["Adj Close_Gold"]].copy()
    data['returns'] = data.pct_change()
    data['log_returns'] = np.log(1 + data['returns'])
    data = data.dropna(how='any')

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'