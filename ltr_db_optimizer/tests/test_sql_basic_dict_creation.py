import pytest
from ltr_db_optimizer.parser.SQLParser import create_query_element

def all_empty(element_list, compare_dict):
    for element in element_list:
        if compare_dict[element]:
            return False
    return True


def test_dict_creation_select():
    query = {'FROM': ['tcph.dbo.SUPPLIER'],
     "SELECT": [{"FIELD": "S_ACCTBAL"}]
    }
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert compare_dict["Tables"][0][0] == "supplier", f"Wrong table in 'Tables'. Expected: 'supplier'. Received: {compare_dict['Tables'][0]}"
    assert len(compare_dict["Tables"][0][1]) == 1, f"Length of fields in 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'][1])}"
    assert compare_dict["Tables"][0][1][0] == "S_ACCTBAL", f"Wrong field in 'Tables'. Expected: 'S_ACCTBAL'. Received: {compare_dict['Tables'][1][0]}"
    
    assert len(compare_dict["Select"]) == 1, f"Length of Tables in 'Select' is not correct. Expected: 1. Received: {len(compare_dict['Select'])}"
    assert compare_dict["Select"][0][0] == "supplier", f"Wrong table in 'Select'. Expected: 'supplier'. Received: {compare_dict['Select'][0]}"
    assert compare_dict["Select"][0][1] == "S_ACCTBAL", f"Wrong field in 'Select'. Expected: 'S_ACCTBAL'. Received: {compare_dict['Select'][1][0]}"
    
    assert all_empty(["Joins", "Aggregation", "Sort", "Filter", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"

def test_dict_creation_select_all():
    query = {'FROM': ['tcph.dbo.REGION']
    }
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert compare_dict["Tables"][0][0] == "region", f"Wrong table in 'Tables'. Expected: 'region'. Received: {compare_dict['Tables'][0]}"
    assert len(compare_dict["Tables"][0][1]) == 0, f"Length of fields in 'Tables' is not correct. Expected: 0. Received: {len(compare_dict['Tables'][1])}"
    
    assert len(compare_dict["Select"]) == 3, f"Length of Tables in 'Select' is not correct. Expected: 3. Received: {len(compare_dict['Select'])}"
    assert all(
        el[0] == "region" for el in compare_dict["Select"]
    ), f"Wrong tables in 'Select'. Expected: 'region'. Received: {compare_dict['Select']}"
    assert all(
        el[1] in ["R_REGIONKEY", "R_NAME", "R_COMMENT"] for el in compare_dict["Select"]
    ), f"Wrong tables in 'Select'. Expected: ['R_REGIONKEY', 'R_NAME', 'R_COMMENT']. Received: {compare_dict['Select']}"
    
    assert all_empty(["Joins", "Aggregation", "Sort", "Filter", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"

    
def test_dict_creation_aggregation_all():
    query = {'FROM': ['tcph.dbo.SUPPLIER'],
             "SELECT": [{'FIELD': '*', 'OPERATOR': 'COUNT'},
                        {'FIELD': 'S_ACCTBAL', 'OPERATOR': 'AVG'}]
    }
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert compare_dict["Tables"][0][0] == "supplier", f"Wrong table in 'Tables'. Expected: 'supplier'. Received: {compare_dict['Tables'][0]}"
    assert len(compare_dict["Tables"][0][1]) == 1, f"Length of fields in 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'][1])}"
    assert compare_dict["Tables"][0][1][0] == "S_ACCTBAL", f"Wrong field in 'Tables'. Expected: 'S_ACCTBAL'. Received: {compare_dict['Tables'][1][0]}"
    
    assert len(compare_dict["Select"]) == 2, f"Length of Tables in 'Select' is not correct. Expected: 2. Received: {len(compare_dict['Select'])}"
    assert all(
        el[0] == "Group By" for el in compare_dict["Select"]
    ), f"Not all elements in 'Select' are of type 'Group By': {compare_dict['Select']}"
    assert all(
        el[1] in ["COUNT(*)", "AVG(S_ACCTBAL)"] for el in compare_dict['Select']
    ), f"Wrong fields in 'Select': {compare_dict['Select']}"
    
    assert compare_dict['Aggregation'] != None, "Aggregation is None"
    assert compare_dict['Aggregation']["Type"] == "All", f"'Type' of 'Aggregation' was expected to be 'All', but is {compare_dict['Aggregation']['Type']}"
    assert len(compare_dict['Aggregation']["Group By"]) == 0, f"Length of 'Group By' in 'Aggregation' should be 0, but is { len(compare_dict['Aggregation']['Group By'])}"
    assert all(
        el[0] in ["COUNT", "AVG"] for el in compare_dict['Aggregation']['Outputs']
    ), f"Wrong operations in 'Aggregation'-'Outputs': {compare_dict['Aggregation']['Outputs']}"
    assert any(el[1] == "supplier" for el in compare_dict['Aggregation']['Outputs']), f"'supplier' was not found in 'Aggregation'-'Outputs': {compare_dict['Aggregation']['Outputs']}" 
    assert any(el[2] in ["*",  "S_ACCTBAL"] for el in compare_dict['Aggregation']["Outputs"]), f"Wrong fields in 'Aggregation'-'Outputs': {compare_dict['Aggregation']['Outputs']}"
    
    assert all_empty(["Joins", "Sort", "Filter", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"
    
    
def test_dict_creation_group_by():
    query = {'FROM': ['tcph.dbo.SUPPLIER'],
         "SELECT": [{'FIELD': '*', 'OPERATOR': 'COUNT'},
                    {'FIELD': 'S_ACCTBAL', 'OPERATOR': 'AVG'}],
         'GROUP BY': ['S_NATIONKEY'],
    }
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert compare_dict["Tables"][0][0] == "supplier", f"Wrong table in 'Tables'. Expected: 'supplier'. Received: {compare_dict['Tables'][0]}"
    assert len(compare_dict["Tables"][0][1]) == 2, f"Length of fields in 'Tables' is not correct. Expected: 2. Received: {len(compare_dict['Tables'][1])}"
    assert all(
        el in ["S_ACCTBAL", "S_NATIONKEY"] for el in compare_dict['Tables'][0][1]
    ), f"Wrong fields in 'Tables'. Expected: ['S_ACCTBAL', 'S_NATIONKEY']. Received: {compare_dict['Tables'][1]}"
    
    assert len(compare_dict["Select"]) == 2, f"Length of Tables in 'Select' is not correct. Expected: 2. Received: {len(compare_dict['Select'])}"
    assert all(el[0] == "Group By" for el in compare_dict["Select"]), f"Not all elements in 'Select' are of type 'Group By': {compare_dict['Select']}"
    assert all(
        el[1] in ["COUNT(*)", "AVG(S_ACCTBAL)"] for el in compare_dict["Select"]
    ), f"Wrong fields in 'Select': {compare_dict['Select']}"
    
    assert compare_dict['Aggregation'] != None, "Aggregation is None"
    assert compare_dict['Aggregation']["Type"] == "Group", f"'Type' of 'Aggregation' was expected to be 'Group', but is {compare_dict['Aggregation']['Type']}"
    assert len(compare_dict['Aggregation']["Group By"]) == 1, f"Length of 'Group By' in 'Aggregation' should be 1, but is { len(compare_dict['Aggregation']['Group By'])}"
    assert compare_dict['Aggregation']["Group By"][0] == ("supplier", "S_NATIONKEY"), f"Wrong 'Group By' in 'Aggregation': {compare_dict['Aggregation']['Group By']}"
    
    assert all(
        el[0] in ["COUNT", "AVG"] for el in compare_dict['Aggregation']['Outputs']
    ), f"Wrong operations in 'Aggregation'-'Outputs': {compare_dict['Aggregation']['Outputs']}"
    assert any(el[1] == "supplier" for el in compare_dict['Aggregation']["Outputs"]), f"'supplier' was not found in 'Aggregation'-'Outputs': {compare_dict['Aggregation']['Outputs']}" 
    assert all(
        el[2] in ["*", "S_ACCTBAL"] for el in compare_dict['Aggregation']['Outputs']
    ), f"Wrong fields in 'Aggregation'-'Outputs': {compare_dict['Aggregation']['Outputs']}"  
    
    assert all_empty(["Joins", "Sort", "Filter", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"
    

def test_dict_creation_filter():
    query = {'FROM': ['tcph.dbo.NATION'],
         'SELECT': [{'FIELD': 'N_NATIONKEY'}],
         'WHERE': [{'FIELD': 'N_REGIONKEY', 'OPERATOR': '=', 'VALUE': '2.0'}],
    }
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert compare_dict["Tables"][0][0] == "nation", f"Wrong table in 'Tables'. Expected: 'nation'. Received: {compare_dict['Tables'][0]}"
    assert len(compare_dict["Tables"][0][1]) == 2, f"Length of fields in 'Tables' is not correct. Expected: 2. Received: {len(compare_dict['Tables'][1])}"
    assert all(
        el in ['N_NATIONKEY', 'N_REGIONKEY'] for el in compare_dict["Tables"][0][1]
    ), f"Wrong fields in 'Tables'. Expected: ['N_NATIONKEY', 'N_REGIONKEY']. Received: {compare_dict['Tables'][1]}"
    
    assert len(compare_dict["Select"]) == 1, f"Length of Tables in 'Select' is not correct. Expected: 1. Received: {len(compare_dict['Select'])}"
    assert compare_dict["Select"][0] == ('nation', 'N_NATIONKEY'), f"Wrong field in 'Select'. Expected: ('nation', 'N_NATIONKEY'). Received: {compare_dict['Select']}"
    
    assert len(compare_dict["Filter"]) == 1, f" Wrong length of 'Filter'. Expected: 1. Received: {len(compare_dict['Filter'])}" 
    assert compare_dict["Filter"][0] == ('=', 'nation', 'N_REGIONKEY', '2.0'), f"Wrong filter. Expected: ('=', 'nation', 'N_REGIONKEY', '2.0'), Received: {compare_dict['Filter'][0]}" 
    
    assert all_empty(["Joins", "Aggregation", "Sort", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"
    
def test_dict_creation_join():
    query = {'FROM': ['tcph.dbo.REGION', 'tcph.dbo.NATION'],
         'SELECT': [{'FIELD': 'R_NAME'}, {'FIELD': 'N_NATIONKEY'}],
         'WHERE_JOIN': [{'FIELD': 'N_REGIONKEY', 'LEFT': 'nation', 'RIGHT': 'region'}]
        }
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 2, f"Length of 'Tables' is not correct. Expected: 2. Received: {len(compare_dict['Tables'])}"
    assert all(
        el[0] in ["region", "nation"] for el in compare_dict['Tables']
    ), f"Wrong tables in 'Tables'. Expected: ['nation', 'region']. Received: {compare_dict['Tables']}"
    assert all(
        len(el[1]) == 2 for el in compare_dict["Tables"]
    ), f"Wrong fields in 'Tables'. Received: {compare_dict['Tables']}"
    
    fields = []
    for table in compare_dict["Tables"]:
        fields.extend(table[1])
    
    assert all(
        f in ['R_NAME', 'R_REGIONKEY', 'N_NATIONKEY', 'N_REGIONKEY'] for f in fields
    ), f"Wrong fields in 'Tables'. Received: {compare_dict['Tables'][1]}"
    
    
    assert len(compare_dict["Select"]) == 2, f"Length of Tables in 'Select' is not correct. Expected: 2. Received: {len(compare_dict['Select'])}"
    assert any(
        el[0]  == "region" for el in compare_dict["Select"]
    ), f"Wrong tables in 'Select'. Expected: ['nation', 'region']. Received: {compare_dict['Select']}"
    assert any(
        el[0]  == "nation" for el in compare_dict["Select"]
    ), f"Wrong tables in 'Select'. Expected: ['nation', 'region']. Received: {compare_dict['Select']}"
    assert all(
        el[1] in ["R_NAME", "N_NATIONKEY"] for el in compare_dict["Select"]
    ), f"Wrong fields in 'Select'. Expected: ['R_NAME', 'N_NATIONKEY']. Received: {compare_dict['Select']}"
    
    assert len(compare_dict["Joins"]) == 1, f"Length of Joins is not correct. Expected: 1. Received: {len(compare_dict['Joins'])}"
    assert (
        compare_dict["Joins"][0] == ('nation', 'N_REGIONKEY', 'region', 'R_REGIONKEY') or compare_dict["Joins"][0] == ('region', 'R_REGIONKEY', 'nation', 'N_REGIONKEY')
    ), f"Join is not correct. Expected: ('nation', 'N_REGIONKEY', 'region', 'R_REGIONKEY'). Received: {compare_dict['Joins'][0]}"
    
    assert all_empty(["Aggregation", "Sort", "Filter", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"
    
def test_dict_creation_sort():
    query = {'FROM': ['tcph.dbo.LINEITEM'],
             'SELECT': [{'FIELD': 'L_TAX'}],
             'ORDER BY': [{'FIELD': 'L_SHIPMODE', 'ORDER': 'DESC'}]}
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert all(
        el[0] in ["lineitem"] for el in compare_dict["Tables"]
    ), f"Wrong tables in 'Tables'. Expected: ['lineitem']. Received: {compare_dict['Tables']}"
    assert len(compare_dict["Tables"][0][1]) == 2, f"Wrong fields in 'Tables'. Received: {compare_dict['Tables']}"
    assert all(
        el in ['L_TAX', 'L_SHIPMODE'] for el in compare_dict['Tables'][0][1]
    ), f"Wrong fields in 'Tables'. Received: {compare_dict['Tables']}"
    
    
    assert len(compare_dict["Select"]) == 1, f"Length of Tables in 'Select' is not correct. Expected: 1. Received: {len(compare_dict['Select'])}"
    assert compare_dict["Select"][0] == ("lineitem", "L_TAX"), f"Wrong fields in 'Select'. Received: {compare_dict['Select'][0]}"
    
    assert len(compare_dict["Sort"]) == 1, f"Length of fields in 'Sort' is not correct. Expected: 1. Received: {len(compare_dict['Sort'])}"
    assert compare_dict["Sort"][0] == ('DESC', 'L_SHIPMODE', 'lineitem'), f"Wrong field in 'Sort'. Expected: ('DESC', 'L_SHIPMODE', 'lineitem'). Received: {compare_dict['Sort'][0]}"
        
    assert all_empty(["Aggregation", "Joins", "Filter", "Top"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"
    
def test_dict_creation_top():
    query = {'FROM': ['tcph.dbo.LINEITEM'],
             'SELECT': [{'FIELD': 'L_TAX'}],
             'TOP': 100}
    compare_dict = create_query_element(query)
    
    assert len(compare_dict["Tables"]) == 1, f"Length of 'Tables' is not correct. Expected: 1. Received: {len(compare_dict['Tables'])}"
    assert all(
        el[0] in ["lineitem"] for el in compare_dict["Tables"]
    ), f"Wrong tables in 'Tables'. Expected: ['nation', 'region']. Received: {compare_dict['Tables']}"
    assert len(compare_dict["Tables"][0][1]) == 1, f"Wrong fields in 'Tables'. Received: {compare_dict['Tables']}"
    assert compare_dict["Tables"][0][1][0] == 'L_TAX', f"Wrong fields in 'Tables'. Received: {compare_dict[v][1]}"

    assert len(compare_dict["Select"]) == 1, f"Length of Tables in 'Select' is not correct. Expected: 1. Received: {len(compare_dict['Select'])}"
    assert compare_dict["Select"][0] == ("lineitem", "L_TAX"), f"Wrong fields in 'Select'. Received: {compare_dict['Select'][0]}"
    
    assert compare_dict["Top"] == 100, f"Number of 'Top' is not correct. Expected: 100. Received: {compare_dict['Top']}"        
    assert all_empty(["Aggregation", "Joins", "Filter", "Sort"], compare_dict), f"There is an non-empty element which should be empty: {compare_dict}"