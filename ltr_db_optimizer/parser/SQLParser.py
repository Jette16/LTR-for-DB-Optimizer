from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

from ltr_db_optimizer.enumeration_algorithm.table_info import TPCHTableInformation
from ltr_db_optimizer.enumeration_algorithm.utils import match_operations
import ltr_db_optimizer.enumeration_algorithm.enumeration_node as nodes

table_info = TPCHTableInformation()



def clear_query_element(query: dict) -> dict:
    """
    This function clears the query and 'deletes' unnecessary elements.
    Example: A sort after field "X" if there is already an equality filter for this field.
    The sort is 'deleted' for the enumeration algorithm, because we do not need to sort it.
    
    Only unnecessary ORDER BY or double FILTER statements are regarded. The other fields are not 'cleared'.
    
    :param query: The query in form of a dictionary which should be cleared.
    :returns: The cleared query in form of a dictionary, the keys are kept! 
    """
    # If there are no filters or sorts: Return the query as it is
    if "ORDER BY" not in query.keys() and "WHERE" not in query.keys():
        return query
    
    # First: Copy everything from the query dict (except for "ORDER BY"/"WHERE" keys) into a new dict
    new_query = {}
    for key in query.keys():
        if key != "ORDER BY" and key != "WHERE":
            new_query[key] = query[key]
    
    # Clear the WHERE part
    if "WHERE" in query.keys():
        filter_fields = {}
        # Group the filters by their field
        for filt in query["WHERE"]:
            if filt["FIELD"] in filter_fields.keys():
                filter_fields[filt["FIELD"]].append((filt["OPERATOR"],filt["VALUE"]))
            else:
                filter_fields[filt["FIELD"]] = [(filt["OPERATOR"],filt["VALUE"])]
                
        # Check for filtered fields with more than one filter. Check if the filters contradict each other. 
        for filt in filter_fields.keys():
            if len(filter_fields[filt]) > 1:
                try:
                    temp_operation = (filter_fields[filt][0][0], filter_fields[filt][0][1])
                    for value in filter_fields[filt][1:]:
                        possible, temp_operation = match_operations(temp_operation[0],
                                                                  temp_operation[1],
                                                                  value[0],
                                                                  value[1])
                        # TODO: If there is a contradiction, the query returns only an empty value
                        if not possible:
                            raise Exception()
                    
                except:
                    continue
            else:
                temp_operation = (filter_fields[filt][0][0], filter_fields[filt][0][1])
            
            if "WHERE" not in new_query.keys():
                new_query["WHERE"] = [{"FIELD":str(filt),
                                       "OPERATOR": temp_operation[0],
                                       "VALUE": temp_operation[1]
                                      }]
            else:
                new_query["WHERE"].append({"FIELD":str(filt),
                                           "OPERATOR": temp_operation[0],
                                           "VALUE": temp_operation[1]
                                          })
    if "ORDER BY" in query.keys():
        filter_fields = []
        if "WHERE" in query:
            filter_fields = [el["FIELD"] for el in query["WHERE"] if el["OPERATOR"] == "="]
        order_by = []
        for order in query["ORDER BY"]:
            # test at first, if the "ORDER BY"-key is in one of the filters
            if order["FIELD"] in filter_fields:
                continue
            # if there is a group to a field which is already filtered
            if "GROUP BY" in query.keys():
                if all([el in filter_fields for el in query["GROUP BY"]]):
                    break
                to_break = False
                for el in query["GROUP BY"]:
                    try:
                        # I don't think that this case is important for more than one group
                        if len(query["GROUP BY"]) == 1:
                            group_table = table_info.match_prefix(el.split("_")[0]+"_")
                            if any([f in filter_fields for f in group_table.field_restrictions]):
                                to_break = True
                    except:
                        continue

                if to_break:
                    continue
                
                
            if "(" not in order["FIELD"]:
                critic_fields = table_info.match_prefix(order["FIELD"].split("_")[0]+"_").field_restrictions
                if critic_fields:
                    if any([el in critic_fields for el in filter_fields]):
                        continue
            
            order_by.append(order)     
        if order_by:
            new_query["ORDER BY"] = order_by
    
    return new_query

def create_query_element(query):
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
    
    # fill tables
    for t in query["FROM"]:
        table_name = t.split(".")[-1].lower()
        table = table_info.get_table(table_name)
        prefix = table.prefix
        
        cols = []
        for key in query.keys():
            if str(key) == "FROM" or str(key) == "TOP":
                continue
            for el in query[key]:
                if str(key) == "GROUP BY":
                    if el.startswith(prefix) and el not in cols:
                        cols.append(el)
                elif str(key) == "WHERE_JOIN":
                    if (table_name == el["LEFT"] or table_name == el["RIGHT"]):
                        temp = prefix+el["FIELD"].split("_")[1]
                        if temp not in cols:
                            cols.append(temp)
                else:
                    if el["FIELD"].startswith(prefix) and el["FIELD"] not in cols:
                        cols.append(el["FIELD"])
        frame["Tables"].append((table_name, None)) # We do not have any aliases
    
    # fill joins
    if "WHERE_JOIN" in query.keys():
        for j in query["WHERE_JOIN"]:
            table_1 = j["LEFT"]
            table_2 = j["RIGHT"]
            field = j["FIELD"].split("_")[1]
            col_1 = table_info.get_table(table_1).prefix + field
            col_2 = table_info.get_table(table_2).prefix + field
            frame["Joins"].append((table_1, col_1.lower(), table_2, col_2.lower()))
            
    # fill aggregation
    if "GROUP BY" in query.keys() or ("SELECT" in query.keys() and any("OPERATOR" in el.keys() for el in query["SELECT"])):
        temp_d = {}
        temp_d["Group By"] = []
        temp_d["Outputs"] = []
        if "GROUP BY" in query.keys():
            temp_d["Type"] = "Group"
            for i in query["GROUP BY"]:
                temp_t = table_info.match_prefix(i.split("_")[0]+"_").table_name
                temp_d["Group By"].append((temp_t.lower(), i.lower()))
        else:
            temp_d["Type"] = "All"
        
        if "ORDER BY" in query.keys():
            for o in query["ORDER BY"]:
                if "(" in o["FIELD"]:
                    operation = o["FIELD"].split("(")[0]
                    field =  o["FIELD"].split("(")[1].split(")")[0]
                    temp_t = None
                    if field != "*":
                        temp_t = table_info.match_prefix(field.split("_")[0]+"_").table_name.lower()
                    temp_d["Outputs"].append((operation.lower(), temp_t, field.lower()))
                    
        for s in query["SELECT"]:
            if "OPERATOR" in s.keys():
                operation = s["OPERATOR"]
                field = s["FIELD"]
                if field == "*":
                    temp_t = None
                else:
                    temp_t = table_info.match_prefix(field.split("_")[0]+"_").table_name.lower()
                temp_d["Outputs"].append((operation.lower(), temp_t, field.lower()))
        
        frame["Aggregation"] = temp_d
        
    # fill sort
    if "ORDER BY" in query.keys():
        for o in query["ORDER BY"]:
            order = o["ORDER"]
            if "(" in o["FIELD"]:
                field =  o["FIELD"]
                temp_t = "Group By"
            else:
                field =  o["FIELD"]
                temp_t = table_info.match_prefix(field.split("_")[0]+"_").table_name
            frame["Sort"].append((order.lower(), field.lower(), temp_t.lower()))
            
    # fill top        
    if "TOP" in query.keys():
        frame["Top"] = query["TOP"]
    
    if "WHERE" in query.keys():
        for w in query["WHERE"]:
            field =  w["FIELD"]
            temp_t = table_info.match_prefix(field.split("_")[0]+"_").table_name
            
            if w["OPERATOR"] == "BETWEEN":                
                op_1 = ">="
                op_2 = "<="
                try:
                    temp_val_1 = float(w["VALUE"][0])
                    temp_val_2 = float(w["VALUE"][1])
                    if temp_val_1 < temp_val_2:
                        val_1 = w["VALUE"][0]
                        val_2 = w["VALUE"][1]
                    else:
                        val_2 = w["VALUE"][0]
                        val_1 = w["VALUE"][1]
                except:
                    val_1 = w["VALUE"][0]
                    val_2 = w["VALUE"][1]
                frame["Filter"].append((op_1, temp_t.lower(), field.lower(), val_1))
                frame["Filter"].append((op_2, temp_t.lower(), field.lower(), val_2))
                continue
                
            operator = w["OPERATOR"]
            if "dateadd" in w["VALUE"]:
                val = w["VALUE"]
                date = val.split("'")[1]
                temp = int(val.split(",")[1])
                if "mm" in val:
                    val = datetime.strptime(date, "%Y-%m-%d") + relativedelta(months=temp)
                elif "yy" in val:
                    val = datetime.strptime(date, "%Y-%m-%d") + relativedelta(years=temp)
                else:
                    continue
                val = val.strftime("%Y-%m-%d")
                val = f"'{val}'"
            else:
                val = w["VALUE"]
            frame["Filter"].append((operator, temp_t.lower(), field.lower(), val))
        
    # fill select
    if "SELECT" in query.keys():
        table_name = ""
        if any("OPERATOR" in el.keys() for el in query["SELECT"]):
            table_name = "Group By"
        for s in query["SELECT"]:
            if "OPERATOR" in s.keys():
                field = s["OPERATOR"]+"("+s["FIELD"]+")"
            else:
                field = s["FIELD"]
                if table_name != "Group By":
                    table_name = table_info.match_prefix(field.split("_")[0]+"_").table_name
            
            frame["Select"].append((table_name.lower(), field.lower()))     

    else: # There is a SELECT *
        for table in frame["Tables"]:
            t = table_info.get_table(table)
            for column in t.get_columns():
                frame["Select"].append((t.table_name.lower(), column.lower()))
            
    alias_dict = {"Fields": {}, "Tables": {}}
    
    return frame, alias_dict

def has_subquery(query):
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
