import os
from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes

class XMLParser:
    entry_to_logical = {
        "table_scan": "Table Scan",
        "index_scan": "Clustered Index Scan",
        "top": "Top",
        "nested_loop_join": "Inner Join",
        "hash_join": "Inner Join",
        "merge_join": "Inner Join",
        "sort": "Sort",
        "hash_aggregate": "Aggregate",
        "compute_scalar": "Compute Scalar",
        "stream_aggregate": "Aggregate",
    }
    entry_to_physical = {
        "table_scan": "Table Scan",
        "index_scan": "Clustered Index Scan",
        "top": "Top",
        "nested_loop_join": "Nested Loops",
        "hash_join": "Hash Match",
        "merge_join": "Merge Join",
        "sort": "Sort",
        "hash_aggregate": "Hash Match",
        "compute_scalar": "Compute Scalar",
        "stream_aggregate": "Stream Aggregate",
    }
    compare_op = {
        "<": "LT",
        "<=": "LE",
        ">": "GT",
        ">=": "GE",
        "=": "EQ",
        "IS": "IS",
        "EQ": "EQ",
        "!=": "NE"
    }
    
    def __init__(self, table_info = None, template_path=None, small_version = False):
        if not template_path:
            self.template_path = "./ltr_db_optimizer/parser/xml_templates/"
        else:
            self.template_path = template_path
        self.templates = self.read_xml_files()
        if table_info is None:
            self.table_info = TPCHTableInformation()
        else:
            self.table_info = table_info
        self.small_version = small_version
        
    def generate_from_graph(self, graph):
        xml = self.templates["header"]
        body, _ = self.generate_xml_part(graph, 0)
        xml = self.replace_body(xml, body)
        xml = xml.replace("'", "&#39;") #escaping ' character
        return self.correct_indentation(xml)
        
    def generate_xml_part(self, graph, node_id):
        xml = self.generate_rel_op(graph, node_id)
        if not self.small_version:
            output_list = self.generate_output_list(graph.output_columns)
        else:
            output_list = "<OutputList />"
        body = eval("self.generate_"+graph.name)(graph)
        node_id += 1
        # generate "own" body
        xml = self.replace_body(xml, output_list + "\n" + body)
        
        if graph.has_children():
            child_body = ""
            for child in graph.get_children()[::-1]:
                temp, node_id = self.generate_xml_part(child, node_id)
                child_body += temp
                child_body += "\n"
            xml = self.replace_body(xml, child_body[:-1])
        return xml, node_id
    
    def generate_stream_aggregate(self, node):
        assert type(node) == nodes.AggregateNode and node.name == "stream_aggregate"
        
        xml = self.templates["stream_aggregate"]
        def_values = self.templates["defined_values"]
        body = ""
        for agg in node.aggregation_operations:
            temp = self.get_column_ref(agg["output_name"]) + "\n"
            if agg["operation"] == "COUNT*":
                temp += self.generate_scalar_operator({"agg_type": agg["operation"],
                                                       "distinct": "false"},
                                                       "aggregate")
            else:
                temp += self.generate_scalar_operator({"agg_type": agg["operation"],
                                                       "distinct": "false",
                                                       "value": ("identifier", {"column": agg["column"]})},
                                                      "aggregate")
            body += self.replace_body(self.templates["defined_value"], temp) + "\n"
        def_values = self.replace_body(def_values, body[:-1])
        xml = xml.replace("####DEFINEDVALUES####", def_values)
        
        if node.aggregation_columns:
            body = ""
            for col in node.aggregation_columns:
                body += self.get_column_ref(col) + "\n"
            body = self.replace_body(self.templates["group_by"], body[:-1])
            xml = xml.replace("####GROUPBY####",body)
        else:
            xml = xml.replace("\n####GROUPBY####","")
        return xml
    
    
    def generate_aggregate(self, agg_dict):
        if agg_dict["agg_type"] == "COUNT*":
            xml = self.templates["aggregate_2"]
        else:
            xml = self.templates["aggregate"]
        xml = xml.replace("####AGGTYPE####", agg_dict["agg_type"])
        xml = xml.replace("####DISTINCT####", agg_dict["distinct"])
        if agg_dict["agg_type"] == "COUNT*":
            return xml
        else:
            body = self.generate_scalar_operator(agg_dict["value"][1], agg_dict["value"][0])
            return self.replace_body(xml, body)
    
    
    def generate_hash_aggregate(self, node):
        assert type(node) == nodes.AggregateNode and node.name == "hash_aggregate"
        
        xml = self.templates["hash"]
        def_values = self.templates["defined_values"]
        body = ""
        for agg in node.aggregation_operations:
            temp = self.get_column_ref(agg["output_name"]) + "\n"
            if agg["operation"] == "COUNT*":
                temp += self.generate_scalar_operator({"agg_type": agg["operation"],
                                                       "distinct": "false"},
                                                       "aggregate")
            else:
                temp += self.generate_scalar_operator({"agg_type": agg["operation"],
                                                       "distinct": "false",
                                                       "value": ("identifier", {"column": agg["column"]})},
                                                      "aggregate")
            body += self.replace_body(self.templates["defined_value"], temp) + "\n"
        def_values = self.replace_body(def_values, body[:-1])
        xml = xml.replace("####DEFINEDVALUES####", def_values)
        
        if node.aggregation_columns:
            hash_keys_build = "" 
            build_residual = self.templates["build_residual"]
            if len(node.aggregation_columns) > 1:
                build_residual = self.replace_body(build_residual, self.templates["scalar_operator"])
                build_residual = self.replace_body(build_residual, self.templates["logical"])
                build_residual = build_residual.replace("####OPERATION####", "AND")
            body = ""
            for col in node.aggregation_columns:
                body += self.generate_scalar_operator({"column": col, "column_2": col, "operator": "IS"},"compare") + "\n"
                hash_keys_build += self.get_column_ref(col) + "\n"
            hash_keys_build = self.replace_body(self.templates["hash_keys_build"], hash_keys_build[:-1])
            build_residual = self.replace_body(build_residual, body[:-1])
            xml = xml.replace("####BUILDRESIDUAL####",build_residual)
        else:
            xml = xml.replace("\n####BUILDRESIDUAL####","")
            hash_keys_build = "<HashKeysBuild />"
            
        xml = xml.replace("\n####HASHKEYSPROBE####","")
        xml = xml.replace("####HASHKEYSBUILD####",hash_keys_build)
        xml = xml.replace("\n####PROBERESIDUAL####","")
        return xml
    
    def generate_convert(self, convert_dict):
        xml = self.templates["convert"]
        xml = xml.replace("####DATATYPE####", convert_dict["datatype"])
        if convert_dict["datatype"] == "decimal":
            xml = xml.replace("####PRECISION####", convert_dict["precision"])
            xml = xml.replace("####SCALE####", convert_dict["scale"])
        else:
            xml = xml.replace("####DATATYPE####", convert_dict["datatype"])
            xml = xml.replace("####DATATYPE####", convert_dict["datatype"])
        xml = xml.replace("####STYLE####", convert_dict["style"])
        xml = xml.replace("####IMPLICIT####", convert_dict["implicit"])
        body = self.generate_scalar_operator(convert_dict["value"][1], convert_dict["value"][0])
        return self.replace_body(xml, body)
    
    def generate_arithmetic(self, arith_dict):
        xml = self.templates["arithmetic"]
        xml = xml.replace("####OPERATION####", arith_dict["operator"])
        body = ""
        for val in arith_dict["values"]:
            body += self.generate_scalar_operator(val[1], val[0])
            body += "\n"
        return self.replace_body(xml, body[:-1])

    def generate_compute_scalar(self, node):
        xml = self.templates["compute_scalar"]
        defined_values = self.templates["defined_values"]
        body = ""
        for op in node.operations:
            def_val = self.templates["defined_value"]
            temp = self.get_column_ref(op[1]["name"]) + "\n"
            if op[0] == "IF_ELSE":
                temp += self.generate_if(op[1])
            elif op[0] == "other":
                temp += self.generate_scalar_operator(op[1]["op"][1],op[1]["op"][0])
            def_val = self.replace_body(def_val, temp) + "\n"
            body += def_val
        defined_values = self.replace_body(defined_values, body[:-1])
        return xml.replace("####DEFINEDVALUES####", defined_values)
    
    def generate_if(self, if_dict):
        # there might be cases where no scalar operator is needed !!!
        xml = self.templates["scalar_operator"]
        xml = self.replace_body(xml, self.templates["if"])
        body = ""
        for key in ["if", "then", "else"]:
            if key == "if":
                temp = self.templates["condition"]
            else:
                temp = self.templates[key]
            body += self.replace_body(temp, self.generate_scalar_operator(if_dict[key][1], if_dict[key][0])) +"\n"
        return self.replace_body(xml, body[:-1]) 
    
    def generate_order_by(self, columns, ascending):
        xml = self.templates["order_by"]
        body = ""
        for idx, col in enumerate(columns):
            temp = self.templates["order_by_column"]
            temp = temp.replace("####ASCENDING####", ascending[idx])
            body += temp.replace("####BODY####", self.get_column_ref(col))
            body += "\n"
        return self.replace_body(xml, body[:-1])
    
    
    def generate_sort(self, node):
        assert type(node) == nodes.SortNode
        # I currently don't regard "DISTINCT"-Queries --> Distinct = 'false'
        xml = self.templates["sort"]
        xml = xml.replace("####DISTINCT####", "false")
        order_by = self.generate_order_by(node.column_list, node.ascending_list)
        xml = xml.replace("####ORDERBY####", order_by)
        return xml
    
    def generate_merge_join(self, node):
        assert type(node) == nodes.JoinNode and node.name == "merge_join"
        
        xml = self.templates["merge"]
        if self.small_version:
            inner_side = ""
            outer_side = ""
            res = ""
        else:
            inner_side = self.generate_inner_side_join_columns(node.right_column)
            outer_side = self.generate_outer_side_join_columns(node.left_column)
            res = self.generate_residual(node.left_column, node.right_column)
        xml = xml.replace("####INNERSIDEJOINCOLUMNS####", inner_side)
        xml = xml.replace("####OUTERSIDEJOINCOLUMNS####", outer_side)
        return xml.replace("####RESIDUAL####", res)
    
    
    def generate_inner_side_join_columns(self, column):
        xml = self.templates["inner_side_join_columns"]
        body = self.get_column_ref(column)
        return self.replace_body(xml, body)
    
    def generate_outer_side_join_columns(self, column):
        xml = self.templates["outer_side_join_columns"]
        body = self.get_column_ref(column)
        return self.replace_body(xml, body)
                          
    
    def generate_hash_join(self, node):
        assert type(node) == nodes.JoinNode and node.name == "hash_join"
        # TODO: with BitmapCreator
        xml = self.templates["hash"]
        def_vals = "<DefinedValues />"
        xml = xml.replace("####DEFINEDVALUES####", def_vals)
        xml = xml.replace("####BUILDRESIDUAL####\n", "") # needed for hash aggregate not for join
        if not self.small_version:
            hash_keys_b = self.generate_hash_keys_build(node.right_column)
            hash_keys_p = self.generate_hash_keys_probe(node.left_column)
            probe_res = self.generate_probe_residual(node.left_column, node.right_column)
        else:
            hash_keys_b = ""
            hash_keys_p = ""
            probe_res = ""
            
        xml = xml.replace("####HASHKEYSBUILD####",hash_keys_b)
        xml = xml.replace("####HASHKEYSPROBE####",hash_keys_p)
        return xml.replace("####PROBERESIDUAL####", probe_res)
    
    def generate_probe_residual(self, left_column, right_column):
        xml = self.templates["probe_residual"]
        body = self.generate_scalar_operator({"column": left_column, "column_2": right_column, "operator": "="},"compare")
        return self.replace_body(xml, body)
    
    def generate_residual(self, left_column, right_column):
        xml = self.templates["residual"]
        body = self.generate_scalar_operator({"column": left_column, "column_2": right_column, "operator": "="},"compare")
        return self.replace_body(xml, body)       
    
    def generate_hash_keys_probe(self, column):
        # TODO There might be more than one column
        xml = self.templates["hash_keys_probe"]
        body = self.get_column_ref(column)
        return self.replace_body(xml, body)
    
    def generate_hash_keys_build(self, column):
        # TODO There might be more than one column
        xml = self.templates["hash_keys_build"]
        body = self.get_column_ref(column)
        return self.replace_body(xml, body)
        
    
    def generate_nested_loop_join(self, node):
        # TODO: until now, we will set "Optimized" always to false
        return self.templates["nested_loops"]
    
    def generate_identifier(self, identifier_dict):
        xml = self.templates["identifier"]
        body = self.get_column_ref(identifier_dict["column"])
        return self.replace_body(xml, body)
        
    
    def generate_compare(self, compare_dict):
        xml = self.templates["compare"]
        xml = xml.replace("####COMPAREOP####", self.compare_op[compare_dict["operator"]])
        body = self.generate_scalar_operator({"column": compare_dict["column"]}, "identifier") + "\n"
        if "value" in compare_dict.keys():
            body += self.generate_scalar_operator({"CONSTVALUE": "("+compare_dict["value"]+")"}, "const")
        elif "column_2" in compare_dict.keys():
            body += self.generate_scalar_operator({"column": compare_dict["column_2"]}, "identifier")
                                   
        return self.replace_body(xml, body)
    
    
    def generate_logical(self, logic_dict):
        xml = self.templates["logical"]
        xml = xml.replace("####OPERATION####", logic_dict["logical"].upper())
        body = ""
        if "filters" in logic_dict.keys():
            for f in logic_dict["filters"]:
                if "logical" in f.keys():
                    body += self.generate_scalar_operator(f, "logical")
                    body += "\n"
                elif "operator" in f.keys():
                    if f["operator"] in self.compare_op.keys():
                        body += self.generate_scalar_operator(f, "compare")
                        body += "\n"
                    # TODO: maybe a second elif to secure that there is no wrong input
                    else:
                        body += self.generate_scalar_operator(f, "intrinsic")
                        body += "\n"
            body = body[:-1]
        elif "values" in logic_dict.keys():
            body = self.generate_scalar_operator(logic_dict["values"][1],logic_dict["values"][0])
        else:
            raise Exception("Unknown operation in generate_logical")
        return self.replace_body(xml, body)
    
    def generate_predicate(self, filter_dict):
        # currently only for <=, <, =, >, >=
        xml = self.templates["predicate"]
        if len(filter_dict["filters"]) > 1:
            body = self.generate_scalar_operator(filter_dict, "logical")
        else:
            if "logical" in filter_dict["filters"][0].keys():
                body = self.generate_scalar_operator(filter_dict["filters"][0], "logical")
            elif filter_dict["filters"][0]["operator"] in self.compare_op.keys():
                body = self.generate_scalar_operator(filter_dict["filters"][0], "compare")
            else:
                body = self.generate_scalar_operator(filter_dict["filters"][0], "intrinsic")
        return self.replace_body(xml, body)
            
    def generate_top(self, node):
        assert type(node) == nodes.TopNode
        val = "("+str(node.top_nr)+")"
        body_1 = self.generate_scalar_operator({"CONSTVALUE": val}, "const", val)
        body_2 = self.templates["top_expression"]
        body = self.replace_body(body_2, body_1) +"\n####BODY####"
        xml = self.templates["top"]
        return self.replace_body(xml, body)
    
    def generate_scalar_operator(self, info_dict, body_type, scalar_string = None):
        if scalar_string:
            xml = self.templates["scalar_operator_2"]
            xml = xml.replace("####SCALARSTRING####", scalar_string)
        else:
            xml = self.templates["scalar_operator"]
        body = eval("self.generate_"+body_type)(info_dict)
        xml = self.replace_body(xml, body)            
        return xml
    
    def generate_intrinsic(self, intrinsic_dict):
        xml = self.templates["intrinsic"]
        xml = xml.replace("####FUNCTIONNAME####", intrinsic_dict["operator"].lower())
        body = self.generate_scalar_operator({"column": intrinsic_dict["column"]},"identifier") + "\n"
        body += self.generate_scalar_operator({"CONSTVALUE": intrinsic_dict["value"]}, "const")
        return self.replace_body(xml, body)
    
    
    def generate_const(self, const_dict):
        xml = self.templates["const"]
        xml = xml.replace("####CONSTVALUE####", const_dict["CONSTVALUE"])
        return xml
    
    def generate_table_scan(self, node):
        xml = self.generate_index_scan(node, True)
        xml = xml.replace(' Index="####INDEX####"', '')
        xml = xml.replace('####INDEXKIND####', 'Heap')
        return xml
        
    def generate_index_scan(self, node, table_scan = False):
        assert type(node) == nodes.ScanNode and len(node.contained_tables) == 1
        
        if node.has_alias():
            table_name = node.get_alias()
        else:
            table_name = node.contained_tables[0]
        
        xml = self.templates["index_scan"]
        #if part_dict["additional_info"]:
            # not done yet, for "Ordered" and so
        if self.small_version or node.is_sorted:
            xml = xml.replace("####ORDERED####", "true")
            if self.small_version:
                xml = xml.replace(' ScanDirection="####SCANDIRECTION####"', '')
            else:
                xml = xml.replace("####SCANDIRECTION####", node.scan_direction)
        else:
            xml = xml.replace("####ORDERED####", "false")
            xml = xml.replace(' ScanDirection="####SCANDIRECTION####"', '')
        database = self.table_info.database
        schema = self.table_info.schema
        xml = xml.replace("####DATABASE####", "["+database+"]")
        xml = xml.replace("####SCHEMA####", "["+schema+"]")
        xml = xml.replace("####TABLE####", "["+table_name.upper()+"]") # upper might not be necessary 
        
        if database == "imdb":
            xml = xml.replace("####ALIAS####", "["+node.contained_tables[0]+"]")
        else:
            xml = xml.replace('Alias="####ALIAS####" ', "")
        if not table_scan:
            xml = xml.replace("####INDEX####", self.table_info.get_table(table_name).get_server_key())
            xml = xml.replace("####INDEXKIND####","Clustered")    
        
        # TODO: Care for seek and Storage as additional info
        xml = xml.replace("####STORAGE####", "RowStore")
        # if there is a seek, replace it
        seek = ""
        if ((not self.small_version) or self.table_info.database == "imdb") and node.filters:
            # for now only handled: <, <=, >=, >
            seek = "\n" + self.generate_predicate(node.filters)
        
        xml = xml.replace("####SEEK####", seek)
        
        columns = ""
        if self.small_version or not len(node.output_columns):
            def_vals = "<DefinedValues />"
        else:
            def_vals = self.templates["defined_values"]
            for col in node.output_columns:
                temp = self.templates["defined_value"]
                temp_body= self.generate_column_reference(col, database, schema, table_name)
                temp = self.replace_body(temp, temp_body)
                columns += temp + "\n"
            def_vals = self.replace_body(def_vals, columns[:-1])
        
        xml = self.replace_body(xml, def_vals)       
        return xml
    
    def generate_column_reference(self, column, database=None, schema=None, table=None):
        if database and schema and table:
            xml = self.templates["column_reference"]
            xml = xml.replace("####DATABASE####", database)
            xml = xml.replace("####SCHEMA####", schema)
            xml = xml.replace("####TABLE####", table)            
        else:
            xml = self.templates["column_reference_2"]
        xml = xml.replace("####COLUMN####", column)
        return xml       
    
    def generate_rel_op(self, node, node_id):
        xml = self.templates["rel_op"]
        
        xml = xml.replace("####NODEID####", str(node_id))
        xml = xml.replace("####ESTIMATEDEXECUTIONMODE####", node.execution_mode)
        
        xml = xml.replace("####AVGROWSIZE####", str(0))
        xml = xml.replace("####ESTIMATECPU####", str(0))
        xml = xml.replace("####ESTIMATEIO####", str(0))
        xml = xml.replace("####ESTIMATEREBINDS####", str(0))
        xml = xml.replace("####ESTIMATEREWINDS####", str(0))
        xml = xml.replace("####ESTIMATEROWS####", str(0))
        xml = xml.replace("####LOGICALOP####", self.entry_to_logical[node.name])
        xml = xml.replace("####PARALLEL####", "false")
        xml = xml.replace("####PHYSICALOP####", self.entry_to_physical[node.name])
        xml = xml.replace("####ESTIMATEDTOTALSUBTREECOST####", str(0))       
        return xml
    
    def generate_output_list(self, output_list):
        if not len(output_list):
            return "<OutputList />"
        
        xml = self.templates["output_list"]
        body = ""
        for col in output_list:
            # I assume that I will not name a calculated column with an underscore inside
            if "_" in col and self.table_info.database != "imdb":
                table_name = self.column_to_tablename(col)
                body += self.generate_column_reference(col, self.table_info.database, self.table_info.schema, table_name)
            else:
                body += self.generate_column_reference(col)
            body += "\n"
        body = body[:-1]
        return self.replace_body(xml, body)
    
    def replace_body(self, body_xml, inside_xml):
        return body_xml.replace("####BODY####", inside_xml)        
        
    def correct_indentation(self, xml):
        correct_xml = ""
        normal_indent = "  "
        xml_split = xml.split("\n")
        indent_count = 0
        
        
        for line in xml_split:
            l = line
            while l.startswith(" "):
                l = l[1:]
            if line.endswith("/>") or line.endswith("?>"):
                indent = normal_indent * indent_count
                correct_xml += indent
                correct_xml += line
                correct_xml += "\n"
            elif line.startswith("</"):
                indent_count -= 1
                indent = normal_indent * indent_count
                correct_xml += indent
                correct_xml += line
                correct_xml += "\n"
            else:
                indent = normal_indent * indent_count
                correct_xml += indent
                correct_xml += line
                correct_xml += "\n"
                indent_count += 1
        return correct_xml
    
    def read_xml_files(self):
        xmls = {}
        for f in os.listdir(self.template_path):
            if not f.endswith(".txt"):
                continue
            with open(self.template_path+f, "r") as file:
                s = ""
                for line in file:
                    s += line
                xmls[f.split(".")[0]] = s
        return xmls
    
    def get_column_ref(self, column):
        if "_" in column and self.table_info.database != "imdb":
            return self.generate_column_reference(column,
                                                  self.table_info.database,
                                                  self.table_info.schema,
                                                  self.column_to_tablename(column))
        else:
            return self.generate_column_reference(column)
    
    def column_to_tablename(self, column):
        table_prefix = column.split("_")[0]+"_"
        table = self.table_info.match_prefix(table_prefix)
        return table.table_name
        