import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes
import ltr_db_optimizer.enumeration_algorithm.utils as utils

then_op = ("const", {"CONSTVALUE": "NULL"})

class Aggregator:
    com_scalar_op = ["AVG", "SUM"]
    
    def __init__(self, sql_query, joiner, is_imdb = False):
        self.aggregation = sql_query["Aggregation"]
        self.joiner = joiner
        
        self.agg_dict = {}
        self.agg_matcher = {}
        self.aggregation_cols = []
        self.has_group_by = True
        if self.has_aggregate():
            self.prepare_aggregate()
        self.is_imdb = is_imdb
        
    def has_aggregate(self):
        return self.aggregation != {}
    
    def get_translated_column(self, column):
        # For cases where the column gets another name in the aggregator or is renamed
        if column in self.agg_matcher.keys():
            return self.agg_matcher[column]
        else:
            return column
        
    def prepare_aggregate(self):
        # prepare in the beginning what kind of aggregation is needed
        if len(self.aggregation["Group By"]):
            self.agg_dict["group_by"] = self.aggregation["Group By"]
        else:
            self.has_group_by = False
            
        self.agg_dict["needed_fields"] = []
        for idx,col in enumerate(self.aggregation["Outputs"]):
            name = "aggregate"+str(idx)
            self.agg_dict["needed_fields"].append((col[0], col[1], col[2], name))
            # renamed column after aggregation
            self.agg_matcher[col[0]+"("+col[2]+")"] = name        
            
    def add_aggregate(self, best_plan):
        if self.has_group_by:
            group_cols = [el[1] for el in self.agg_dict["group_by"]]
        # test if aggregate is necessary
        # e.g., it does not seem to be necessary if we call group by on a unique column
        if self.has_group_by and len(best_plan.unique_columns) > 0:
            if all([el in group_cols for el in best_plan.unique_columns]):
                self.joiner.append_restricted("count(*)")
                return self.insert_compute_scalar_without_aggregate(best_plan)
        
        aggregates = []
        # get the plan with an appended sort for the stream aggregate
        if self.has_group_by:
            sorted_plan = self.return_sorted_plan(best_plan, group_cols)
        else:
            sorted_plan = best_plan
            group_cols = []
        
        aggregate_operations = []
        
        compute_scalar = []
        compute_scalar_output = []
        already_computed = {}
        counter = 0
        
        # append the needed operation for every needed field
        
        for operation in self.agg_dict["needed_fields"]:
            
            # some operations like avg and sum require a succeeding compute scalar operator 
            if operation[0] in self.com_scalar_op:
                count_str = str(counter)
                if operation[2] in already_computed.keys():
                    count_str = already_computed[operation[2]]
                
                column_name = "tempcount"+count_str
                
                if_op = self.get_if(column_name)
                
                if operation[0].lower() == "avg":
                    else_op = self.get_arithmetic(column_name)
                elif operation[0].lower() == "sum":
                    else_op = self.get_sum(column_name)
                else:
                    raise Exception("Unknown operator for compute_scalar")
                    
                compute_scalar.append(self.get_if_else(if_op, then_op, else_op, operation[3]))
                compute_scalar_output.append(operation[3])
                counter += 1

                if operation[2] not in already_computed.keys():
                    aggregate_operations.append(self.get_aggregate_operation(column_name, "SUM", operation[2]))
                    aggregate_operations.append(self.get_aggregate_operation(column_name, "COUNT_BIG", operation[2]))
                    # ensure no double calculation 
                    already_computed[operation[2]] = count_str
                    self.agg_matcher["tempsum"+count_str] = operation[2]
                    
                self.agg_matcher[operation[3]] = ["tempsum"+count_str, "tempcount"+count_str]
                
            # the operator for count(*) has another name in SQL Server than the     
            elif operation[0].lower() == "count" and operation[2] == "*":
                aggregate_operations.append(self.get_aggregate_operation(operation[3], "COUNT*"))
                self.agg_matcher[operation[3]] = None
            else:
                aggregate_operations.append(self.get_aggregate_operation(operation[3], operation[0], operation[2]))
                self.agg_matcher[operation[3]] = operation[2]
                
        # somehow hash aggregates are not possible for imdb, not sure why. This needs further investigations.
        if not self.is_imdb:
            aggregates.append(nodes.AggregateNode("hash_aggregate", group_cols, aggregate_operations,
                                                  name = "hash_aggregate", left_child = best_plan,
                                                  contained_tables = best_plan.contained_tables))
            
        aggregates.append(nodes.AggregateNode("stream_aggregate", group_cols, aggregate_operations,
                                              name = "stream_aggregate", left_child = sorted_plan, is_sorted = True,
                                              contained_tables = sorted_plan.contained_tables, sorted_columns = group_cols))
                          
        # append compute scalar if needed
        if len(compute_scalar):
            aggregates = self.insert_compute_scalar(aggregates, compute_scalar)
        return aggregates
                          
    
    def return_sorted_plan(self, plan, columns):
        # calculate a sorted plan if the plans isn't sorted already
        if not self.has_group_by:
            return plan
        if not all([o in plan.sorted_columns for o in columns]):
            return nodes.SortNode(columns, ["true"]*len(columns), name = "sort",
                                  left_child = plan, is_sorted = True,
                                  contained_tables = plan.contained_tables,
                                  sorted_columns = columns)                         
        else:
            return plan
    
    def get_arithmetic(self, column_name):
        d = ("arithmetic", {"operator": "DIV",
                            "values": [("identifier", {"column": column_name}),
                                       ("convert", {"datatype": "decimal",
                                                    "precision": "19",
                                                    "scale": "0",
                                                    "style": "0",
                                                    "implicit": "true",
                                                    "value": ("identifier", {"column": column_name}) 
                                                   })]
                           })
        return d
    
    def get_if(self, column_name):
        d = ("compare", {"operator": "EQ", 
                         "column": column_name,
                         "value": "0"})
        return d
    
    def get_sum(self, column_name):
        return ("identifier", {"column": column_name})
    
    def get_aggregate_operation(self, column_name, operation, column=None):
        if not column:
            return {"output_name": column_name, "operation": operation}
        return {"output_name": column_name, "column": column, "operation": operation}
    
    def get_if_else(self, if_op, then_op, else_op, name):
        return ("IF_ELSE", {"if": if_op, "then": then_op, "else": else_op, "name": name})

    ######################## Compute Scalar Operations

    def insert_compute_scalar(self, aggregates, compute_scalar):
        result = []
        for agg in aggregates:
            result.append(nodes.ComputeScalarNode(compute_scalar, name="compute_scalar", left_child=agg, is_sorted=agg.is_sorted,
                                                  contained_tables = agg.contained_tables, sorted_columns = agg.sorted_columns,
                                                  unique_columns = agg.unique_columns))
        return result
    
    def insert_compute_scalar_without_aggregate(self, plan):
        # some rules for constructing a valid compute scalar node for SQL Server
        compute_scalar = []
        counter = 0
        for operation in self.agg_dict["needed_fields"]:
            op = ()
            count_str = str(counter)
            if operation[0].lower() in ["avg", "sum"]:
                op = ("other", {"op":("convert", {"datatype": "decimal",
                                  "precision": "19",
                                  "scale": "0",
                                  "style": "0",
                                  "implicit": "true",
                                  "value": ("identifier", {"column": operation[2]}) 
                                 }), "name":operation[3]})
                compute_scalar.append(op)
                self.agg_matcher[operation[3]] = [operation[2]]
            elif operation[0].lower() in ["min", "max"]:
                op = ("other", {"op": ("identifier", {"column": operation[2]}), "name":operation[3]})
                compute_scalar.append(op)
                self.agg_matcher[operation[3]] = [operation[2]]
            elif operation[0].lower() == "count":
                if_op = ("logical", {"logical": "IS NULL", "values": self.get_const_value("0")})

                then_op = self.get_const_value("0")
                else_op = self.get_const_value("1")
                
                compute_scalar.append(self.get_if_else(if_op, then_op, else_op, operation[3]))
                self.agg_matcher[operation[3]] = [None]
            else:
                raise Exception("Unknown operator for compute_scalar")
            
        aggregates = self.insert_compute_scalar([plan], compute_scalar)
        return aggregates    
        
    def get_const_value(self, value):
        return ("const", {"CONSTVALUE": value})