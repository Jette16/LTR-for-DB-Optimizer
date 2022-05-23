import networkx as nx 
import pyodbc

from ltr_db_optimizer.enumeration_algorithm.dpccp import DPccp
from ltr_db_optimizer.enumeration_algorithm.joiner import Joiner
from ltr_db_optimizer.enumeration_algorithm.aggregator import Aggregator
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes
from ltr_db_optimizer.model.featurizer_graph import FeatureExtractorGraph
   
from ltr_db_optimizer.parser.xml_parser import XMLParser
from ltr_db_optimizer.parser.SQLParser import to_sql

class EnumerationAlgorithm(DPccp):
    def __init__(self, sql_query, table_info, model, sql_text, top_k = 1, alias_dict = {}, comparison=False):
        self.path = model
        super().__init__(model = model, joiner = Joiner(sql_query, alias_dict, table_info), comparison=comparison)
        self.top_k = top_k
        self.sql_query = sql_query
        self.sql_text = sql_text
        self.to_bfs_graph()
        self.table_info = table_info
        self.aggregator = Aggregator(sql_query, self.joiner, self.table_info.database == "imdb")
        
        self.connection = 'dummyConnection' # see https://github.com/mkleehammer/pyodbc/wiki/Drivers-and-Driver-Managers#connecting-to-a-database
        self.saved_rows = {}
        self.featurizer = FeatureExtractorGraph()
        
        self.regard_subquery = False
        self.alias_dict = alias_dict
        
    def find_best_plan(self):
        # Check if there is a subquery in our regular query, call a new instance and handle the subquery first
        if not self.regard_subquery:
            if self.sql_query["Subquery"] is not None:
                sub_enum = EnumerationAlgorithm(self.sql_query["Subquery"], self.table_info,
                                                self.path, "", self.top_k, self.alias_dict)
                sub_enum.regard_subquery = True
                self.best_subquery = sub_enum.find_best_plan()
                self.joiner.set_subquery(self.best_subquery)
                
        # only start enumeration if we have at least one join
        if len(self.sql_query["Joins"]) > 0:
            best_plan = self.enumerate(name_in_data=True)
        else:
            assert len(self.sql_query["Tables"]) == 1
            best_plan = self.joiner.get_scan(self.sql_query["Tables"][0][0])
        
        # add aggregations, sorts and top statements
        best_plan = self.add_additional_nodes(best_plan)
        
        if self.regard_subquery:
            return best_plan
        
        # reduce the final plans to one plan
        best_plan = self.reduce(best_plan, True)[0]
        return self.to_xml(best_plan)
        
        
    def to_bfs_graph(self):
        """
        Using the join information in sql_query, a networkx.Graph is constructed with the relations as nodes
        and the join-relations as edges. To simplify the enumeration, a breadth-first search is applied and the
        node names are changed to increasing integers.
        """        
        self.graph = nx.Graph()
        self.graph.add_nodes_from([table[0].lower() for table in self.sql_query["Tables"]])
        self.graph.add_edges_from([(edge[0], edge[2]) for edge in self.sql_query["Joins"]])
        self.graph = nx.bfs_tree(self.graph, source=list(self.graph.nodes)[0])
        self.graph = nx.convert_node_labels_to_integers(self.graph, label_attribute="old_name")
        
    def add_additional_nodes(self, best_plans):
        temp_result = []
        
        # add aggregates if needed
        for plan in best_plans:
            if self.aggregator.has_aggregate():
                best_plan = self.aggregator.add_aggregate(plan)
                temp_result.extend(best_plan)                
            else:
                temp_result.append(plan)
        
        result = []
        
        # append sorts and top statements if needed
        for plan in temp_result:
            best_plan = plan
            if len(self.sql_query["Sort"]):
                plan.query_encoding[0] = 1
                # check if it is already sorted or if the whole table has only one row --> no sort needed
                if ((not all([o[1] in best_plan.sorted_columns for o in self.sql_query["Sort"]]) and
                    not all([self.joiner.is_restricted(s[1]) for s in self.sql_query["Sort"]])) or 
                    any([s[0] == "desc" for s in self.sql_query["Sort"]])
                   ):
                    columns = []
                    ascending = []
                    for order in self.sql_query["Sort"]:
                        columns.append(self.aggregator.get_translated_column(order[1]))
                        asc = "true" if order[0] == "asc" else "false"
                        ascending.append(asc)
                    best_plan = nodes.SortNode(columns, ascending, name = "sort", left_child = best_plan,
                                               is_sorted = True, contained_tables = best_plan.contained_tables,
                                               sorted_columns = columns)
            # Insert a top node if there is a top statement in the query
            if self.sql_query["Top"] is not None:
                best_plan = nodes.TopNode(self.sql_query["Top"], name = "top", left_child = best_plan,
                                           is_sorted = best_plan.is_sorted, contained_tables = best_plan.contained_tables,
                                           sorted_columns = best_plan.sorted_columns)
            result.append(best_plan) 
        # somehow, SQL Server seems to have problems with the batch mode, which is why we change all execution modes to row mode
        # This should be investigated more
        if self.table_info.database == "imdb":
            temp_result = []
            for plan in result:
                temp_result.append(plan.down_propagate())
            return temp_result
        return result
    
    def resolve_output_columns(self, plan, output_columns=None, sort = False):
        """
        For the XML, find the output columns of every node
        """
        if output_columns is None:
            output_columns = []
            for s in self.sql_query["Select"]:
                output_columns.append(self.aggregator.get_translated_column(s[1]))
        plan.set_output_columns(output_columns)
        # True for everything except Scans
        if plan.has_children():
            # True 
            if not plan.has_right_child():
                columns = []
                if type(plan) == nodes.ComputeScalarNode or type(plan) == nodes.AggregateNode:
                    for col in output_columns:
                        matched = self.aggregator.get_translated_column(col)
                        if type(matched) == list:
                            columns.extend(matched)
                        elif matched is None or matched.startswith("tempcount"):
                            continue
                        else:
                            columns.append(matched)
                    if plan.name == "stream_aggregate":
                        sort = True
                    plan.left_child = self.resolve_output_columns(plan.left_child, set(columns), sort)
                    
                else: # For Top and Sort (?)
                    if plan.name == "sort":
                        sort = False
                    plan.left_child = self.resolve_output_columns(plan.left_child, set(output_columns), sort)
            else:
                # Handle Joins
                right_columns = []
                left_columns = []  
                for col in output_columns:
                    if "." in col:
                        table = col.split(".")[0]
                    else:
                        pre = col.split("_")[0]+"_"
                        table = self.table_info.match_prefix(pre).table_name
                    if table in plan.left_child.contained_tables:
                        left_columns.append(col)
                    elif table in plan.right_child.contained_tables:
                        right_columns.append(col)
                    else:
                        raise Exception(f"Table {table} not found when trying to split the columns.")
                if plan.left_column not in left_columns:
                    left_columns.append(plan.left_column)
                if plan.right_column not in right_columns:
                    right_columns.append(plan.right_column)
                if plan.name == "merge_join":
                    sort = True
                plan.left_child = self.resolve_output_columns(plan.left_child, set(left_columns), sort)
                plan.right_child = self.resolve_output_columns(plan.right_child, set(right_columns), sort)
        else:
            plan.is_sorted = sort
        return plan
        
    def prepare_plans(self, plans, last = False):
        # get the SQL of the subplan and get the plan encoding of the trees 
        if last:
            sql = self.sql_text            
        else:
            sql = to_sql(plans[0], self.table_info)
        feat_plans, rows = self.plans_to_feature_vecs(sql, plans, last)
        query_enc = plans[0].get_query_encoding()
        return query_enc, feat_plans
    
    def plans_to_feature_vecs(self, sql, plans, last):
        # get the plan encodings
        parser = XMLParser(table_info=self.table_info, small_version=True)
        rows = 0
        result_featurized = []
        conn = pyodbc.connect(self.connection)
        cursor = conn.cursor()
        for plan in plans:
            need_sql = self.featurizer.append_cost(plan)
            # to reduce the number of calls of SQL Server, we store the already calculated
            # numbers of rows, i.e., we do not need to call SQL Server for every query
            if need_sql:
                xml = parser.generate_from_graph(plan)
                plan_sql = sql + " OPTION (RECOMPILE, USE PLAN N'"+xml+"')"
                cursor.execute("SET SHOWPLAN_XML ON")
                try:
                    rows = cursor.execute(plan_sql).fetchall()
                except:
                    raise Exception
                cost_plan = rows[0][0]
                self.featurizer.match_cost_plan(plan, cost_plan)
            temp, rows = self.featurizer.featurize_node(plan)
            result_featurized.append(temp)
        conn.close()
        return result_featurized, rows
        
    def to_xml(self, plan):
        parser = XMLParser(table_info=self.table_info, small_version=True)
        sql = self.sql_text
        xml = parser.generate_from_graph(plan)
        plan_sql = sql + " OPTION (RECOMPILE, USE PLAN N'"+xml+"')"
        return plan_sql
        
        
        
        