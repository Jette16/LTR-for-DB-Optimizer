from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
import numpy as np

query_encoding_length = 6
min_vec = np.array([0.0, 0.0, 0.0, 1.0, 5.0, 5.0])
max_vec = np.array([1.0, 1.0, 7.0, 4.801740e+08, 6.001215e+06, 6.001215e+06])

class EnumerationNode:
    def __init__(self, name = "", left_child = None, right_child = None,
                 is_sorted = False, contained_tables = [], sorted_columns = [],
                 unique_columns = [], output_columns = [], estimated_rows = None, # change back to None 
                 query_encoding = None, execution_mode = "Batch", estimated_cost = 0, # only needed for bao
                 **kwargs):                 
        self.name = name
        self.left_child = left_child
        self.right_child = right_child
        self.is_sorted = is_sorted
        contained_tables.sort()
        self.contained_tables = contained_tables
        self.sorted_columns = sorted_columns
        self.unique_columns = unique_columns
        self.output_columns = output_columns
        self.estimated_rows = estimated_rows
        self.estimated_cost = estimated_cost
        if query_encoding:
            self.query_encoding = query_encoding
        else:
            self.calculate_query_encoding()
        self.execution_mode = execution_mode
        if self.has_left_child():
            self.child_est = self.left_child.has_rows()
            if self.has_right_child():
                self.child_est = self.child_est and self.right_child.has_rows()
        self.id = "_".join(self.contained_tables)+"_"+self.name.split("_")[-1]
        self.featurized_plan = None
        
    def get_right_child(self):
        return self.right_child
    
    def get_left_child(self):
        return self.left_child
    
    def has_right_child(self):
        return not (self.right_child is None)
    
    def has_left_child(self):
        return not (self.left_child is None)
    
    def has_children(self):
        return self.has_left_child()
    
    def get_children(self):
        if not self.right_child is None:
            return [self.left_child, self.right_child]
        elif self.has_left_child():
            return [self.left_child]
        return []
    
    def is_sorted(self):
        return self.is_sorted
    
    def contains(self, table_name):
        return table_name in self.contained_tables
    
    def contains_one(self, columns):
        return any([self.contains(col) for col in columns])
    
    def set_output_columns(self, output_columns):
        self.output_columns = output_columns
    
    def has_filters(self):
        return False
    
    def has_rows(self):
        return self.estimated_rows is not None
    
    def get_query_encoding(self):
        return self.query_encoding
    
    def calculate_query_encoding(self):
        self.query_encoding = [0] * query_encoding_length
    
    def normalize_query(self):
        self.query_encoding = list((np.array(self.query_encoding)-min_vec)/(max_vec - min_vec))
    
    def set_estimated_rows(self, rows):
        self.estimated_rows = rows
        self.query_encoding[3] = rows
        self.normalize_query()
        
    def down_propagate(self):
        copy_node = EnumerationNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        if copy_node.has_left_child():
            copy_node.left_child = copy_node.left_child.down_propagate()
        if copy_node.has_right_child():
            copy_node.right_child = copy_node.right_child.down_propagate()
        return copy_node
    
    def set_featurized_plan(self, plan):
        self.featurized_plan = plan
    
    def get_featurized_plan(self):
        return self.featurized_plan
        
    def has_featurized_plan(self):
        return self.featurized_plan is not None
    
            
class JoinNode(EnumerationNode): 
    def __init__(self, join_type, left_column, right_column, **kwargs):
        super().__init__(**kwargs)
        self.join_type = join_type
        self.left_column = left_column
        self.right_column = right_column
        self.contained_tables = self.right_child.contained_tables + self.left_child.contained_tables
        if join_type in ["nested_loop_join", "merge_join"]:
            self.execution_mode = "Row"
            
        
    def calculate_query_encoding(self):
        # new: orderby, group_by, nr_joins, estimated_rows, max. relation, min relation
        vector_left = self.left_child.get_query_encoding()
        vector_right = self.right_child.get_query_encoding()
        vector = [0] * query_encoding_length
        vector[0] = vector_left[0] + vector_right[0]
        vector[1] = vector_left[1] + vector_right[1]
        vector[1] = vector_left[2] + vector_right[2] + 1
        vector[4] = vector_left[4] if vector_left[4] > vector_right[4] else vector_right[4]
        vector[5] = vector_left[5] if vector_left[5] < vector_right[5] else vector_right[5]
        vector[3] = self.estimated_rows
        self.query_encoding = vector
        
    def down_propagate(self):
        copy_node = JoinNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        if copy_node.has_left_child():
            copy_node.left_child = copy_node.left_child.down_propagate()
        if copy_node.has_right_child():
            copy_node.right_child = copy_node.right_child.down_propagate()
        return copy_node
        
class AggregateNode(EnumerationNode):
    def __init__(self, aggregation_type, aggregation_columns, aggregation_operations,  **kwargs):
        super().__init__(**kwargs)
        self.aggregation_type = aggregation_type
        self.aggregation_columns = aggregation_columns
        self.aggregation_operations = aggregation_operations
        if aggregation_type == "stream_aggregate":
            self.execution_mode = "Row"
    
    def calculate_query_encoding(self):
        # new: orderby, group_by, nr_joins, estimated_rows, max. relation, min relation, nr_columns
        vector = self.left_child.get_query_encoding()
        vector[1] = 1
        vector[3] = self.estimated_rows
        self.query_encoding = vector
        
    def down_propagate(self):
        copy_node = AggregateNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        if copy_node.has_left_child():
            copy_node.left_child = copy_node.left_child.down_propagate()
        if copy_node.has_right_child():
            copy_node.right_child = copy_node.right_child.down_propagate()
        return copy_node        
    
    
class ComputeScalarNode(EnumerationNode):
    def __init__(self, operations, **kwargs):
        super().__init__(**kwargs)
        self.operations = operations
    
    def calculate_query_encoding(self):
        # new: orderby, group_by, nr_joins, estimated_rows, max. relation, min relation, nr_columns
        self.query_encoding = self.left_child.get_query_encoding()
        self.query_encoding[3] = self.estimated_rows
        
    def down_propagate(self):
        copy_node = ComputeScalarNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        if copy_node.has_left_child():
            copy_node.left_child = copy_node.left_child.down_propagate()
        if copy_node.has_right_child():
            copy_node.right_child = copy_node.right_child.down_propagate()
        return copy_node        
             
class SortNode(EnumerationNode):
    def __init__(self, column_list, ascending_list, **kwargs):
        super().__init__(**kwargs)
        self.column_list = column_list
        self.ascending_list = ascending_list
        if self.left_child.has_rows():
            self.estimated_rows = self.left_child.estimated_rows
    
    def calculate_query_encoding(self):
        # new: orderby, group_by, nr_joins, estimated_rows, max. relation, min relation
        self.query_encoding = self.left_child.query_encoding
        
    def down_propagate(self):
        copy_node = SortNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        if copy_node.has_left_child():
            copy_node.left_child = copy_node.left_child.down_propagate()
        if copy_node.has_right_child():
            copy_node.right_child = copy_node.right_child.down_propagate()
        return copy_node        
    
class ScanNode(EnumerationNode):
    def __init__(self, scan_type, filters = None, execution_mode = "Batch", scan_direction = "FORWARD",
                 alias = None, table_info = TPCHTableInformation(), **kwargs):
        self.alias = alias
        self.table_info = table_info
        super().__init__(**kwargs)
        self.scan_type = scan_type
        self.filters = filters
        self.scan_direction = scan_direction
        self.execution_mode = execution_mode
    
    def has_filters(self):
        return self.filters is not None
    
    def calculate_query_encoding(self):
        if self.has_alias():
            relation = self.table_info.get_table(self.alias).row_count
        else:
            relation = self.table_info.get_table(self.contained_tables[0]).row_count
        # new: orderby, group_by, nr_joins, estimated_rows, max. relation, min relation, nr_columns
        self.query_encoding = [0] * query_encoding_length
        self.query_encoding[4] = relation
        self.query_encoding[5] = relation
        self.query_encoding[3] = self.estimated_rows
    
    def down_propagate(self):
        copy_node = ScanNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        return copy_node 
    
    def has_alias(self):
        return self.alias is not None
    
    def get_alias(self):
        return self.alias
        
    
class TopNode(EnumerationNode):
    def __init__(self, top_nr, **kwargs):
        super().__init__(**kwargs)
        self.top_nr = top_nr
        
    def down_propagate(self):
        copy_node = TopNode(**self.__dict__) 
        copy_node.execution_mode = "Row"
        copy_node.estimated_rows = None
        if copy_node.has_left_child():
            copy_node.left_child = copy_node.left_child.down_propagate()
        if copy_node.has_right_child():
            copy_node.right_child = copy_node.right_child.down_propagate()
        return copy_node      
    
        