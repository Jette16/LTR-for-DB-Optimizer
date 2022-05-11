from typing import Union

import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes
from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation as TableInfo


class Joiner:
    
    def __init__(self, sql_query, alias_dict, table_info = TableInfo()):
        self.table_info = table_info
        self.sql_query = sql_query
        self.filter_for_table = {}
        self.prepare_filters()
        if self.table_info == "imdb":
            self.restricted = []
        else:
            self.restricted = self.get_restricted_columns()
        self.subquery = None
        self.alias_dict = alias_dict
        
    def get_join_possibilities(
        self, 
        left_children: Union[nodes.EnumerationNode, list], 
        right_children: Union[nodes.EnumerationNode, list]
    ):
        if type(left_children) == nodes.EnumerationNode:
            left_children = [left_children]
        if type(right_children) == nodes.EnumerationNode:
            right_children = [right_children]
        
        result_nodes = []
        
        tables = right_children[0].contained_tables + left_children[0].contained_tables
        left_column, right_column = self.get_join_columns(left_children[0], right_children[0])
        unique_columns = self.calculate_unique_cols(left_column, right_column, left_children[0].unique_columns, right_children[0].unique_columns)
        
        for right in right_children:
            for left in left_children:
                
                right_child = right
                left_child = left
                row_version_right = right_child if right_child.execution_mode == "Row" else right_child.down_propagate()
                row_version_left = left_child if left_child.execution_mode == "Row" else left_child.down_propagate()
                left_sorted, right_sorted, sorted_columns = self.get_sorted_versions(row_version_left, row_version_right, left_column, right_column)
                # test if needs sort for merge join
                is_restricted = right_column in self.restricted or left_column in self.restricted
                for i in range(2): 
                    result_nodes.append( 
                        nodes.JoinNode("nested_loop_join", left_column, right_column, 
                                       name = "nested_loop_join", left_child = row_version_left, 
                                       right_child = row_version_right, is_sorted = False,
                                       unique_columns = unique_columns, contained_tables = tables)
                    )
                    if not is_restricted:
                        result_nodes.append(
                        nodes.JoinNode("merge_join", left_column, right_column, 
                                       name = "merge_join", left_child = left_sorted, 
                                       right_child = right_sorted, is_sorted = True,
                                       sorted_columns = sorted_columns, unique_columns = unique_columns,
                                       contained_tables = tables)
                        )
                        if self.table_info.database == "imdb":
                            result_nodes.append(
                                nodes.JoinNode("hash_join", left_column, right_column, 
                                               name = "hash_join", left_child = row_version_left, 
                                               right_child = row_version_right, is_sorted = False, 
                                               unique_columns = unique_columns, contained_tables = tables)
                            )
                            
                        else:
                            result_nodes.append(
                                nodes.JoinNode("hash_join", left_column, right_column, 
                                               name = "hash_join", left_child = left_child, 
                                               right_child = right_child, is_sorted = False, 
                                               unique_columns = unique_columns, contained_tables = tables)
                            )
                        
                        
                        # do also Hash and Merge Join
                    
                    # swap variables
                    left_child, right_child = right_child, left_child
                    row_version_right, row_version_left = row_version_left, row_version_right
                    left_column, right_column = right_column, left_column
                    left_sorted, right_sorted = right_sorted, left_sorted
        return result_nodes
    
    
    def get_join_columns(self, left_child, right_child):
        for join in self.sql_query["Joins"]:
            left_table = join[0]
            right_table = join[2]
            prepend_left = ""
            prepend_right = ""
            if right_table in self.alias_dict["Tables"].keys():
                prepend_right = right_table+"."
            if left_table in self.alias_dict["Tables"].keys():
                prepend_left = left_table+"."
            if left_child.contains_one([right_table, left_table]) and right_child.contains_one([left_table, right_table]):
                col_1 = prepend_left+join[1]
                col_2 = prepend_right+join[3]
                if left_child.contains(left_table):
                    return col_1, col_2
                else:
                    return col_2, col_1
                
    def get_restricted_columns(self):
        columns = []
        for filt in self.sql_query["Filter"]:
            if filt[0] == "=":
                if self.table_info.database == "imdb":
                    columns.append(filt[1]+"."+filt[2])
                else:
                    columns.append(filt[2])
        return self.table_info.get_further_reduced_columns(columns, self.sql_query["Joins"])
    
    def is_restricted(self, column):
        return column in self.restricted
    
    def append_restricted(self, column):
        self.restricted.append(column)
    
    
    def get_sorted_versions(self, left_child, right_child, left_column, right_column):
        left_sorted = None
        right_sorted = None
        if left_child.is_sorted and left_column in left_child.sorted_columns:
            left_sorted = left_child
        else:
            left_sorted = nodes.SortNode([left_column], ["true"], name = "sort",
                                         left_child = left_child, is_sorted = True,
                                         contained_tables = left_child.contained_tables,
                                         sorted_columns = [left_column], 
                                         unique_columns = left_child.unique_columns,
                                         execution_mode = "Row")            
        if right_child.is_sorted and right_column in right_child.sorted_columns:
            right_sorted = right_child
        else:
            right_sorted = nodes.SortNode([right_column], ["true"], name = "sort",
                                         left_child = right_child, is_sorted = True,
                                         contained_tables = right_child.contained_tables,
                                         sorted_columns = [right_column],
                                         unique_columns = right_child.unique_columns,
                                         execution_mode = "Row")
            
        
        sorted_columns = [right_column, left_column]
        return left_sorted, right_sorted, sorted_columns
    
    def calculate_unique_cols(self, right_col, left_col, right_unique_cols, left_unique_cols):
        # I currently assume that there is only one "unique" column
        # That is pretty ugly, I hope I can make it more beautiful
        if len(right_unique_cols) == 0 and left_col not in left_unique_cols:
            return left_unique_cols
        if len(left_unique_cols) == 0 and right_col not in right_unique_cols:
            return right_unique_cols
        if len(right_unique_cols) == 1 and right_col in right_unique_cols and not left_col in left_unique_cols:
            return left_unique_cols
        if len(left_unique_cols) == 1 and left_col in left_unique_cols and not right_col in right_unique_cols:
            return right_unique_cols
        if len(right_unique_cols) > 1 and right_col in right_unique_cols and left_col in left_unique_cols:
            return right_unique_cols
        if len(left_unique_cols) > 1 and right_col in right_unique_cols and left_col in left_unique_cols:
            return left_unique_cols
        return []
        
    ############# Handle scan nodes
    def get_scan(self, table_name):
        if table_name.lower() == "subquery" or (table_name in self.alias_dict["Tables"].keys() and self.alias_dict["Tables"][table_name].lower() == "subquery" ):
            return self.subquery
        table_name = table_name.lower()
        filt = self.filter_for_table[table_name] if table_name in self.filter_for_table.keys() else None
        alias = None
        if table_name in self.alias_dict["Tables"].keys():
            alias = self.alias_dict["Tables"][table_name]
            sorted_columns = table_name+"."+self.table_info.get_table(alias).get_first_key()
            unique_columns = [table_name+"."+temp for temp in self.table_info.get_table(alias).get_keys()]
            has_keys = self.table_info.get_table(alias).has_keys() 
        else:
            sorted_columns = self.table_info.get_table(table_name).get_first_key()
            unique_columns = self.table_info.get_table(table_name).get_keys()
            has_keys = self.table_info.get_table(table_name).has_keys() 
        if has_keys:
            return [nodes.ScanNode("index_scan", filt, alias = alias, table_info = self.table_info,
                                   name = "index_scan", is_sorted = True, 
                                  contained_tables = [table_name], sorted_columns = sorted_columns,
                                  unique_columns = unique_columns)]
        else:
            return [nodes.ScanNode("table_scan", filt, alias = alias, table_info = self.table_info,
                                   name = "table_scan", contained_tables = [table_name],
                                   unique_columns = unique_columns)]
    
    def prepare_filters(self):
        for filt in self.sql_query["Filter"]:
            if filt[0] == "in":
                temp_dict = {
                    "logical": "or",
                    "filters": []                    
                }
                if len(filt) == 5:
                    operator = filt[4]
                else:
                    operator = "="
                
                for i in filt[3][1:-1].split(", "):
                    temp_i = {"operator": operator,
                              "column": filt[2],
                              "value": i}
                    temp_dict["filters"].append(temp_i)
                    
                if filt[1] not in self.filter_for_table.keys():
                    self.filter_for_table[filt[1]] = {
                        "logical": "and",
                        "filters": [temp_dict]                    
                    }
                else:
                    self.filter_for_table[filt[1]]["filters"].append(temp_dict)
                continue
                
            if filt[1] not in self.filter_for_table.keys():
                self.filter_for_table[filt[1]] = {
                    "logical": "and",
                    "filters": [{"operator": filt[0],
                                "column": filt[2],
                                "value": filt[3]}]                    
                }
            else:
                temp_f = {"operator": filt[0],
                          "column": filt[2],
                          "value": filt[3]}
                self.filter_for_table[filt[1]]["filters"].append(temp_f)
    
    def set_subquery(self, subquery):
        self.subquery = subquery
            