from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.utils import match_operations
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes

table_info = TPCHTableInformation()


def has_subquery(query):
    """
    Check if a SQL Query has a subquery
    """
    splits = query.lower().split("select ")
    if len(splits) < 3:
        return False
    for sp in splits:
        if len(sp) == 0 or not sp[-1].isalpha():
            continue
        else:
            return False
    return True

def extract_subquery(query):
    """
    Extract a subquery from a SQL query
    """
    # Regards only one 
    splits = re.split(r"(select)", query, flags=re.IGNORECASE)
    #print(splits)
    toggle = True
    main_query = ""
    subquery = ""
    curr_bracket_count = 0
    for sp in splits:
        if not toggle:
            main_query += sp
            if len(sp) > 0 and sp.strip()[-1] == "(":
                curr_bracket_count += 1
                toggle = True
                main_query = main_query.strip()[:-1] + "subquery"
        else:
            if sp.lower() == "select":
                subquery += sp
            else:
                for letter in sp:
                    if toggle:
                        if letter == "(":
                            curr_bracket_count += 1
                        elif letter == ")":
                            curr_bracket_count -= 1
                        if curr_bracket_count == 0:
                            toggle = False
                        else:
                            subquery += letter
                    else:
                        main_query += letter
    return main_query, subquery

def get_date_add(val):
    date = val.split("'")[1]
    temp = int(val.split(",")[1])
    if "mm" in val:
        val = datetime.strptime(date, "%Y-%m-%d") + relativedelta(months=temp)
    elif "yy" in val:
        val = datetime.strptime(date, "%Y-%m-%d") + relativedelta(years=temp)
    else:
        return
    val = val.strftime("%Y-%m-%d")
    return f"'{val}'"

def deal_case(statement, alias_dict):
    """
    Deal if case end statements in SQL Queries
    """
    to_replace = ""
    replace_with = ""
    end = ("case", "end")
    regex = r"(\b(?:{})\b)".format("|".join(end))
    split = re.split(regex, statement, flags=re.IGNORECASE)
    toggle = False
    for s in split:
        if not toggle:
            if s.lower() == "case":
                toggle = True
                to_replace += s
        else:
            if s.lower() == "end":
                toggle = False
            to_replace += s
            # to be changed (maybe with regex find alphanum?)
            if len(s.split(" ")) > 1 and replace_with == "":
                for temp in s.split(" "):
                    if "_" in temp or temp in alias_dict["Fields"].keys():
                        replace_with = alias_dict["Fields"][temp] if temp in alias_dict["Fields"].keys() else temp
                        break
    return to_replace, replace_with            

def deal_or(query):
    """
    Handle or's in queries' WHERE
    """
    or_dict = {}
    
    or_statement = ""
    and_statement = ""
    bracket_counter = 0
    temp_or_statement = ""
    for idx, char in enumerate(query):
        if bracket_counter == 0:
            if char == "(":
                bracket_counter += 1
            else:
                and_statement += char
        else:
            if char == ")":
                bracket_counter -= 1
                if bracket_counter == 0:
                    if " or " in temp_or_statement:
                        if or_statement == "":
                            or_statement = temp_or_statement
                        else:
                            or_statement += " or " + temp_or_statement
                    else:
                        and_statement += "("+temp_or_statement+")"
                    temp_or_statement = ""
            elif char == "(":
                bracket_counter += 1
            else:
                temp_or_statement += char

    split_at = [" and ", "or"] # append or later on
    split_regex = r"\b(?:{})\b".format("|".join(split_at))
    statements = re.split(split_regex, or_statement, flags=re.IGNORECASE)
    operators = [" in ", "<=", ">=", "=", " between ", "<", ">", "like", "!="]
    op_regex = r"((?:{}))".format("|".join(operators))
    for idx, s in enumerate(statements):
        full = re.split(op_regex, s, flags=re.IGNORECASE)
        if full[1].strip().lower() == "=":
            if "_" in full[0] or "." in full[0]:
                field = full[0].strip()
                val = full[2].strip()
                if "dateadd" in val:
                    val = get_date_add(val)
            elif "_" in full[2]:
                field = full[2].strip()
                val = full[0].strip()
                if "dateadd" in val:
                    val = get_date_add(val)
            if field in or_dict.keys():
                or_dict[field].append(val)
            else:
                or_dict[field] = [val]
        elif full[1].strip().lower() == "like":
            if "." in full[0]:
                field = full[0].strip()
                val = full[2].strip()
            if field in or_dict.keys():
                or_dict[field].append((val, "like"))
            else:
                or_dict[field] = [(val, "like")]
    or_list = []
    for key in or_dict.keys():
        if "." in key:
            table = key.split(".")[0]
            field = key.split(".")[1]
        else:
            field = key
            table = table_info.match_from_column(field).table_name
        if all(type(k) == tuple for k in or_dict[key]):
            values = "("+" ,".join((k[0] for k in or_dict[key]))+")"
            or_list.append(("in", table, field, values, or_dict[key][0][1]))
        else:
            or_list.append(("in", table, field, "("+" ,".join(or_dict[key])+")"))
    # operator (in), table (n1), field (n1.name), value ([....])  
    return and_statement, or_list


def from_sql(sql, temp_table_info = None):
    """
    Pretty basic and not pretty SQL to dict parser. There are still some edge cases that are not covered.
    
    """
    global table_info
    if temp_table_info is not None:
        table_info = temp_table_info
    
    buzzwords = ["sum(", "avg(", "min(", "max(", "count("]
    
    def handle_order_by(sql):
        if "order by" in sql.keys():
            result = []
            orders = sql["order by"].split(",")
            for o_full in orders:
                order = "desc" if " desc" in o_full else "asc"
                o = o_full.strip().split(" ")[0] # to remove asc and desc
                if o in alias_dict["Fields"].keys():
                    o = alias_dict["Fields"][o]
                if "(" in o:
                    field = o.strip().split("(")[1].split(")")[0]
                    temp_t = "Group By"                
                else:
                    field = o.strip().split(" ")[0]
                    if "." in field:
                        temp_t = field.split(".")[0]
                        field = field.split(".")[1]
                    else:
                        temp_t = table_info.match_from_column(field).table_name
                result.append((order, field, temp_t))
            return result
        return []
        
    def handle_group_by(sql):
        temp_d = {}
        temp_d["Group By"] = []
        temp_d["Outputs"] = []
        temp_d["Type"] = "All"
        if "group by" in sql.keys():
            temp_d["Type"] = "Group"
            for gb in sql["group by"].split(","):
                g = gb.strip()
                if g in alias_dict["Fields"].keys():
                    g = alias_dict["Fields"][g]
                if "." in g:
                    table = g.split(".")[0]
                    g = g.split(".")[1]
                else:
                    table = table_info.match_from_column(g).table_name
                temp_d["Group By"].append((table, g))
                
        if "select" in sql.keys() and any(b in sql["select"] for b in buzzwords):
            for s in sql["select"].split(","):
                temp = s.strip()
                if any([k in temp for k in buzzwords]):
                    if "case when" in temp.lower():
                        to_replace, replace_with = deal_case(temp, alias_dict)
                        temp = temp.replace(to_replace, replace_with)
                    # TODO wont work for more complicated things
                    operation = temp.split("(")[0]
                    field = temp.split("(")[1].split(")")[0]
                    if field == "*":
                        temp_t = None
                    elif "." in field:
                        temp_t = field.split(".")[0]
                        field = field.split(".")[1]
                    else:
                        if field in alias_dict["Fields"].keys():
                            field = alias_dict["Fields"][field]
                        temp_t = table_info.match_from_column(field).table_name 
                    temp_d["Outputs"].append((operation, temp_t, field))
                else:
                    continue
                
        if "order by" in sql.keys() and any("(" in el for el in sql["order by"].split(",")):
            for o in sql["order by"].split(","):
                order = o.strip()
                if "(" in order:
                    operation = order.split("(")[0]
                    field = order.split("(")[1].split(")")[0]
                    if field == "*":
                        temp_t = None
                    else:
                        temp_t = table_info.match_from_column(field).table_name 
                    temp_d["Outputs"].append((operation, temp_t, field))
        if len(temp_d["Outputs"]) == 0 and len(temp_d["Group By"]) == 0:
            return {}
        return temp_d
    
    def handle_select(sql):
        assert "select" in sql.keys()
        result = []
        table_name = ""
        for s in sql["select"].split(","):
            if any([b in s for b in buzzwords]):
                table_name = "Group By"
                break
        for sp in sql["select"].split(","):
            if "datepart" in sp.lower():
                continue
            s = sp.strip()
            if ")" in s and not "(" in s: # ugly but for datepart
                s = s.replace(")","")
            if s == "*":
                for table in sql["from"].split(","):
                    t = table_info.get_table(table)
                    for column in t.get_columns():
                        result.append((t.table_name, column))
            if "case when " in s.lower():
                to_replace, replace_with = deal_case(s, alias_dict)
                s = s.replace(to_replace, replace_with)
            if (" as " in s.lower()) or len(s.split(" ")) > 1:
                if " as " in s.lower():
                    s = re.split(r" as ", s, flags=re.IGNORECASE)
                else:
                    s = s.split(" ")
                alias_dict["Fields"][s[1]] = s[0]
                s = s[0]
            if table_name != "Group By":
                if "." in s:
                    table_name = s.split(".")[0]
                    s = s.split(".")[1]
                else:
                    table_name = table_info.match_from_column(s).table_name
            result.append((table_name, s))
        return result
    
    def handle_from(sql):
        assert "from" in sql.keys()
        # To do hnalde usbquery
        result = []
        for f in sql["from"].split(","):
            temp = f.strip().split(" ")
            table = temp[0].split(".")[-1]
            if len(temp) == 1:
                result.append((table, None))
            elif len(temp) == 2:
                alias_dict["Tables"][temp[1]] = table
                result.append((temp[1], table))
            elif len(temp) == 3 and temp[1] == "as":
                alias_dict["Tables"][temp[2]] = table
                result.append((temp[2], table))
            else:
                print(temp)
                raise Exception("Unknown FROM split")
        return result
    
    def handle_top(sql): # Done
        if "top" in sql.keys():
            return sql["top"].strip()
        
    def handle_where(sql):
        where_joins = []
        wheres = []
        subquery_joins = []
        if "where" in sql.keys():
            # add more operators later
            operators = [" in ", "<=", ">=", "=", " between ", "<", ">", " like", "!=", " not", " is "]
            op_regex = r"((?:{}))".format("|".join(operators))
            
            and_query = sql["where"]
            if " or " in sql["where"]:
                and_query, or_list = deal_or(sql["where"])
                wheres.extend(or_list)
            
            
            split_at = ["and"] # append or later on
            split_regex = r"\b(?:{})\b".format("|".join(split_at))
            statements = re.split(split_regex, and_query, flags=re.IGNORECASE)
            between = False
            for idx, s in enumerate(statements):
                full = re.split(op_regex, s, flags=re.IGNORECASE)
                if len(full) == 1:
                    if between:
                        between = False
                    continue
                if between:
                    between = False
                    continue
                if len(full) > 3 and (any( f == " not" for f in full)):
                    temp_full = []
                    for idx, f in enumerate(full):
                        if len(f) == 0 or f == " not":
                            continue
                        temp_full.append(f)
                    full = temp_full
                assert len(full) == 3
                if ("_" in full[0] and "_" in full[-1]) or ("." in full[0] and "." in full[-1] and "=" == full[1]):
                    if "." in full[0].strip():
                        table_1 = full[0].strip().split(".")[0]
                        column_1 = full[0].strip().split(".")[1]
                    else:
                        column_1 = full[0].strip()
                        table_1 = table_info.match_from_column(column_1).table_name
                        
                    if "." in full[-1].strip():
                        table_2 = full[-1].strip().split(".")[0]
                        column_2 = full[-1].strip().split(".")[1]
                    else:
                        column_2 = full[-1].strip()
                        table_2 = table_info.match_from_column(column_2).table_name
                    where_joins.append((table_1, column_1, table_2, column_2))
                else:
                    if full[1].strip().lower() == "between":
                        if "." in full[0]:
                            field = full[0].strip().split(".")[1]
                            table = full[0].strip().split(".")[0]
                        else:
                            field = full[0].strip()
                            table = table_info.match_from_column(field).table_name
                        wheres.append((">=", table, field, full[-1].strip()))
                        wheres.append(("<=", table, field, statements[idx+1].strip()))
                        between = True
                        continue
                        
                    operator = full[1].strip()
                    if "_" in full[0] or "." in full[0]:
                        field = full[0].strip()
                        if "." in field:
                            table = field.split(".")[0]
                            field = field.split(".")[1]
                        else:
                            table = table_info.match_from_column(field).table_name
                        val = full[2].strip()
                        if "dateadd" in val:
                            val = get_date_add(val)
                            
                    elif "_" in full[2] or "." in full[2]:
                        field = full[2].strip()
                        if "." in field:
                            table = field.split(".")[0]
                            field = field.split(".")[1]
                        else:
                            table = table_info.match_from_column(field).table_name
                        val = full[0].strip()
                        if "dateadd" in val:
                            val = get_date_add(val)
                    else:
                        raise Exception("Weird")
                    wheres.append((operator, table, field, val))                   
        return where_joins, wheres, subquery_joins

        
    alias_dict = {"Fields": {}, "Tables": {}}
    
    frame = {
        "Tables": [], 
        "Joins": [],
        "Aggregation": {},
        "Sort": [],
        "Top": None,
        "Filter": [],
        "Select": [],
        "Subquery": None
    }
    sql = " ".join(sql.lower().split())
    if has_subquery(sql):
        query, subquery_text = extract_subquery(sql)
        subquery, alias_dict = from_sql(subquery_text)
    else:
        query = sql
        subquery = None
    
    sql_statements = ("select", "where", "order by", "group by", " top", "from")
    regex = r"(\b(?:{})\b)".format("|".join(sql_statements))
    res = re.split(regex, query, flags=re.IGNORECASE)
    res_dict = {}
    top_found = False
    toggle = False
    for idx,r in enumerate(res):
        if r.lower() in sql_statements:
            if r.lower() == "select" and res[idx+2].lower() == " top":
                top_found = True
                res_dict["select"] = " ".join(res[idx+3].split(" ")[2:])
                continue
            elif r.lower() == " top":
                res_dict["top"] = res[idx+1].split(" ")[1]
                toggle = True
                continue
            toggle = True
            res_dict[r.lower()] = res[idx+1]
        if toggle:
            toggle = False
    frame["Tables"] = handle_from(res_dict)
    frame["Select"] = handle_select(res_dict)
    frame["Joins"], frame["Filter"], frame["Subquery_Joins"] = handle_where(res_dict)
    frame["Aggregation"] = handle_group_by(res_dict)
    frame["Sort"] = handle_order_by(res_dict)
    frame["Top"] = handle_top(res_dict)
    frame["Select"] = handle_select(res_dict)
    frame["Subquery"] = subquery
    return frame, alias_dict

def to_sql(execution_plan, temp_table_info = None):
    """
    The purpose is to extract the SQL query out of an execution plan. 
    Currently, we only do this for subplan, i.e. it only handles join, filter, select and from
    """    
    global table_info
    if temp_table_info is not None:
        table_info = temp_table_info
        
    s_select = "SELECT * "
    s_from = "FROM "
    s_where = "WHERE "
    
    curr_node = execution_plan
    on_hold = []
    while True:
        if type(curr_node) == nodes.JoinNode:
            on_hold.append(curr_node.get_right_child())
            if s_where != "WHERE ":
                s_where = s_where + " AND "
            s_where = s_where + curr_node.left_column + "=" + curr_node.right_column
            curr_node = curr_node.get_left_child()
        elif type(curr_node) == nodes.SortNode:
            curr_node = curr_node.get_left_child()
        else:
            if s_from != "FROM ":
                s_from += ", "
            if curr_node.has_alias():
                s_from += (table_info.database+"."+table_info.schema+"."+curr_node.get_alias())
                s_from += " " + curr_node.contained_tables[0]
            else:
                s_from += (table_info.database+"."+table_info.schema+"."+curr_node.contained_tables[0])
            
            if curr_node.has_filters():
                if curr_node.has_alias():
                    alias = curr_node.contained_tables[0]+"."
                else:
                    alias = ""
                for f in curr_node.filters["filters"]:
                    
                    if s_where != "WHERE ":
                        s_where += " AND "
                    # currently, only "IN" statements have these 
                    if "logical" in f.keys(): 
                        s_where += (alias + f["filters"][0]["column"] + " IN " + "(")
                        for sub_f in f["filters"]:
                            s_where += sub_f["value"]+ ", "
                        s_where = s_where[:-2] +") "
                    else:
                        s_where += (alias + f["column"] + " " + f["operator"] + " " + f["value"] + " ")
            if on_hold:
                curr_node = on_hold.pop(0)
            else:
                break
    return s_select + s_from + " " + s_where   
