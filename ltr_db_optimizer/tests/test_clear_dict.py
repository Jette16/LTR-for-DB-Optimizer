import pytest
from ltr_db_optimizer.parser.SQLParser import clear_query_element

def test_no_changes():
    query = {'FROM': ['tcph.dbo.NATION',
                      'tcph.dbo.REGION',
                      'tcph.dbo.SUPPLIER',
                      'tcph.dbo.CUSTOMER'],
             'SELECT': [{'FIELD': '*', 'OPERATOR': 'COUNT'},
                        {'FIELD': 'S_ACCTBAL', 'OPERATOR': 'AVG'},
                        {'FIELD': 'S_ACCTBAL', 'OPERATOR': 'SUM'},
                        {'FIELD': 'C_ACCTBAL', 'OPERATOR': 'SUM'},
                        {'FIELD': 'C_MKTSEGMENT'},
                        {'FIELD': 'N_REGIONKEY'}],
             'GROUP BY': ['C_MKTSEGMENT'],
             'WHERE_JOIN': [{'FIELD': 'R_REGIONKEY', 'LEFT': 'region', 'RIGHT': 'nation'},
                            {'FIELD': 'S_NATIONKEY', 'LEFT': 'supplier', 'RIGHT': 'nation'},
                            {'FIELD': 'C_NATIONKEY', 'LEFT': 'customer', 'RIGHT': 'nation'}]
            }
    new_query = clear_query_element(query)
    assert new_query == query, f"There should be no additional/missing element when clearing query {query}."