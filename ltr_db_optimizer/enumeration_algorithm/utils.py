from itertools import chain, combinations

from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation

# Found on stackoverflow: https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def extend_column_list(columns, tables):
        """
        Extend a list of columns with columns transferring the same info using a list of tables.
        E.g.: columns = [O_CUSTKEY],  tables = [customer, orders] --> result = [O_CUSTKEY, C_CUSTKEY]
        """
        table_info = TPCHTableInformation()
        result_columns = columns
        for column in columns:
            col_name = column.split("_")[1]
            for table in tables:
                table_obj = table_info.get_table(table)
                matched = table_obj.has_column(col_name)
                if matched and matched not in result_columns:
                    result_columns.append(matched)

        return result_columns
    
def match_operations(op_1, value_1, op_2, value_2):
    op_matcher = {
        "=": "equal",
        "<": "less",
        ">": "greater",
        ">=": "greater_equal",
        "<=": "less_equal",
        "IN": "in",
        "LIKE": "like",
        "BETWEEN": "between"
    }
    
    op_1_matched = op_matcher[op_1]
    op_2_matched = op_matcher[op_2]
    
    # try to match the values type:
    try:
        if not type(value_1) == list:
            value_1 = float(value_1)
        else:
            value_1 = [float(value_1[0]), float(value_1[1])]
        if not type(value_2) == list:
            value_2 = float(value_2)
        else:
            value_2 = [float(value_2[0]), float(value_2[1])]
    except:
        value_1 = value_1
        value_2 = value_2
    
    if op_1_matched <= op_2_matched:
        return eval("match_"+op_1_matched+"_"+op_2_matched)(value_1, value_2)
    else:
        return eval("match_"+op_2_matched+"_"+op_1_matched)(value_2, value_1)