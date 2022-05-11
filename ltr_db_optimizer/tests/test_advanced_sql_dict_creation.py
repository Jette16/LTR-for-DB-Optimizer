import pytest
from ltr_db_optimizer.parser.SQLParser import create_query_element

def test_four_tables():
    query = {'FROM': ['tcph.dbo.NATION',
                      'tcph.dbo.REGION',
                      'tcph.dbo.SUPPLIER',
                      'tcph.dbo.CUSTOMER'],
             'WHERE': [{'FIELD': 'N_REGIONKEY', 'OPERATOR': '=', 'VALUE': '2.0'},
                       {'FIELD': 'S_ACCTBAL', 'OPERATOR': '<=', 'VALUE': '7270.635'}],
             'SELECT': [{'FIELD': '*', 'OPERATOR': 'COUNT'},
                        {'FIELD': 'S_ACCTBAL', 'OPERATOR': 'AVG'},
                        {'FIELD': 'S_ACCTBAL', 'OPERATOR': 'SUM'},
                        {'FIELD': 'C_ACCTBAL', 'OPERATOR': 'SUM'},
                        {'FIELD': 'C_MKTSEGMENT'},
                        {'FIELD': 'N_REGIONKEY'}],
             'GROUP BY': ['C_MKTSEGMENT'],
             'WHERE_JOIN': [{'FIELD': 'R_REGIONKEY', 'LEFT': 'region', 'RIGHT': 'nation'},
                            {'FIELD': 'S_NATIONKEY', 'LEFT': 'supplier', 'RIGHT': 'nation'},
                            {'FIELD': 'C_NATIONKEY', 'LEFT': 'customer', 'RIGHT': 'nation'}],
             'TOP': 70}
    
    compare_dict = create_query_element(query)
    
    ## Tests for Tables
    assert len(compare_dict["Tables"]) == 4, f"Wrong length of 'Tables'. Expected: 4. Received: {len(compare_dict['Tables'])}"
    assert all(
        el[0] in ["nation", "region", "supplier", "customer"] for el in compare_dict["Tables"]
    ), f"Wrong tables in 'Tables'. Expected: ['nation', 'region', 'customer', 'supplier']. Received: {compare_dict['Tables']}"
    
    fields = []
    for el in compare_dict["Tables"]:
        fields.extend(el[1]) 
    assert all(
        f in ['N_REGIONKEY', 'N_NATIONKEY', 'R_REGIONKEY', 'S_ACCTBAL', 'S_NATIONKEY', 'C_ACCTBAL', 'C_MKTSEGMENT', 'C_NATIONKEY'] for f in fields
    ), f"Wrong fields in 'Tables': {compare_dict['Tables']}"
    
    # Tests for Select
    assert len(compare_dict["Select"]) == 6, f"Wrong length of 'Select'. Expected: 6. Received: {len(compare_dict['Select'])}"
    assert all(
        el[0] == "Group By" for el in compare_dict["Select"]
    ), f"Wrong tables in 'Select'. Expected: 'Group By'. Received: {len(compare_dict['Select'])}"
    assert all(
        el[1] in ["COUNT(*)","AVG(S_ACCTBAL)", "SUM(S_ACCTBAL)", "SUM(C_ACCTBAL)", "C_MKTSEGMENT", "N_REGIONKEY"] for el in compare_dict["Select"]
    ), f"Wrong fields in 'Select'. Received: {len(compare_dict['Select'])}"
    
    # Tests for Joins
    assert len(compare_dict["Joins"]) == 3, f"Wrong length of 'Joins'. Expected: 3. Received: {len(compare_dict['Joins'])}"
    assert any(
        el == ('region', 'R_REGIONKEY', 'nation', 'N_REGIONKEY') or el == ('nation', 'N_REGIONKEY', 'region', 'R_REGIONKEY') for el in compare_dict["Joins"]
    ), f"Join ('region', 'R_REGIONKEY', 'nation', 'N_REGIONKEY') was not found: {compare_dict['Joins']}"
    assert any(
        el == ('supplier', 'S_NATIONKEY', 'nation', 'N_NATIONKEY') or el == ('nation', 'N_NATIONKEY', 'supplier', 'S_NATIONKEY') for el in compare_dict["Joins"]
    ), f"Join ('supplier', 'S_NATIONKEY', 'nation', 'N_NATIONKEY') was not found: {compare_dict['Joins']}"
    assert any(
        el == ('customer', 'C_NATIONKEY', 'nation', 'N_NATIONKEY') or el == ('nation', 'N_NATIONKEY', 'customer', 'C_NATIONKEY') for el in compare_dict["Joins"]
    ), f"Join ('customer', 'C_NATIONKEY', 'nation', 'N_NATIONKEY') was not found: {compare_dict['Joins']}"
    
    #Test for Sort
    assert compare_dict["Sort"] == [], f"'Sort' is expected to be empty. Received: {compare_dict['Sort']}"
    
    # Test for Top
    assert compare_dict["Top"] == 70, f"Wrong number in 'Top'. Expected: 70. Received: {compare_dict['Top']}"
    
    # Test for Filter
    assert len(compare_dict["Filter"]) == 2, f"Wrong length of 'Filter'. Expected: 2. Received: {len(compare_dict['Filter'])}"
    assert any(
        el == ('=', 'nation', 'N_REGIONKEY', '2.0') for el in compare_dict["Filter"]
    ),  f"Filter ('=', 'nation', 'N_REGIONKEY', '2.0') not found. Received: {len(compare_dict['Filter'])}"
    assert any(
        el == ('<=', 'supplier', 'S_ACCTBAL', '7270.635') for el in compare_dict["Filter"]
    ),  f"Filter ('<=', 'supplier', 'S_ACCTBAL', '7270.635') not found. Received: {len(compare_dict['Filter'])}"
    
    # Tests for Group
    assert compare_dict["Aggregation"]["Type"] == "Group", f"Aggregation-Type was expected to be 'Group' but it is {compare_dict['Aggregation']['Type']}"
    assert len(compare_dict["Aggregation"]["Group By"]) == 1, f"Length of Group By was expected to be 1 but it is {len(compare_dict['Aggregation']['Group By'])}"
    assert compare_dict["Aggregation"]["Group By"][0] == ('customer', 'C_MKTSEGMENT'), f"Group By was exptected to be ('customer', 'C_MKTSEGMENT'), but it is {compare_dict['Aggregation']['Group By']}"
    assert len(compare_dict["Aggregation"]["Outputs"]) == 4, f"Length of Outputs was expected to be 4 but it is {len(compare_dict['Aggregation']['Outputs'])}"
    
    