import copy

class TPCHTableInformation:
    schema = "dbo"
    database = "tpch" # I know there is a typo
    
    def __init__(self):
        self.matcher = {}
        self.matcher["lineitem"] = self.Tables("lineitem", 
                                               "l_",
                                               ["l_orderkey", "l_partkey",
                                                "l_suppkey", "l_linenumber",
                                                "l_quantity", "l_extendedprice",
                                                "l_discount", "l_tax",
                                                "l_returnflag", "l_linestatus",
                                                "l_shipdate", "l_commitdate",
                                                "l_receiptdate", "l_shipinstruct",
                                                "l_shipmode", "l_comment"],
                                               ["l_orderkey", "l_linenumber"],
                                               ["PK__LINEITEM__DD1C9C94D3616509"], 
                                               is_restricting = {"l_orderkey": ["o_orderkey"],
                                                                 "l_partkey": ["p_partkey"],
                                                                 "l_suppkey": ["s_suppkey"],
                                                                },
                                              row_count = 6001215)
        self.matcher["nation"] = self.Tables("nation", 
                                             "n_",
                                             ["n_nationkey", "n_name",
                                              "n_regionkey", "n_comment"],
                                             ["n_nationkey"],
                                             ["PK__NATION__AF64455CF27143A4"],
                                             field_restrictions = ["n_nationkey",
                                                                   "s_nationkey",
                                                                   "c_nationkey"],
                                             is_restricting = {"n_nationkey": ["r_regionkey", "n_name",
                                                                               "n_regionkey", "n_comment",
                                                                               "s_nationkey", "c_nationkey"],
                                                               "n_name": ["n_nationkey", "n_regionkey",
                                                                          "n_comment", "r_regionkey"],
                                                               "n_regionkey": ["r_regionkey"]
                                                              },
                                              row_count = 25
                                            )
        self.matcher["orders"] = self.Tables("orders", 
                                             "o_",
                                             ["o_orderkey", "o_custkey",
                                              "o_orderstatus", "o_totalprice",
                                              "o_orderdate", "o_orderpriority",
                                              "o_clerk", "o_shippriority",
                                              "o_comment"],
                                             ["o_orderkey"],
                                             ["PK__ORDERS__AAA6619D7C63EC05"],
                                             field_restrictions= ["l_orderkey",
                                                                  "o_orderkey"
                                                                 ],
                                             is_restricting = {"o_orderkey": ["o_custkey", "o_orderstatus",
                                                                              "o_totalprice", "o_orderdate",
                                                                              "o_orderpriority", "o_clerk",
                                                                              "o_shippriority", "o_comment",
                                                                              "l_orderkey"],
                                                               "o_custkey": ["c_custkey"]
                                                              },
                                              row_count = 1500000)
        self.matcher["customer"] = self.Tables("customer", 
                                               "c_",
                                    ["c_custkey",
                                     "c_name",
                                     "c_address",
                                     "c_nationkey",
                                     "c_phone",
                                     "c_acctbal",
                                     "c_mktsegment",
                                     "c_comment"],
                                    [],
                                    [],
                                    is_restricting = {"c_custkey": ["c_name", "c_address",
                                                                    "c_nationkey", "c_phone",
                                                                    "c_acctbal", "c_mktsegment",
                                                                    "c_comment", "o_custkey"],
                                                      "c_nationkey": ["n_nationkey"]
                                                     },
                                              row_count = 150000)
        self.matcher["part"] = self.Tables("part", 
                                           "p_",
                                    ["p_partkey", "p_name", "p_mfgr", "p_brand",
                                     "p_type", "p_size", "p_container",
                                     "p_retailprice", "p_comment"],
                                    ["p_partkey"],
                                    ["PK__PART__7FC1E95F40C7D867"],
                                          field_restrictions= ["p_partkey",
                                                               "ps_partkey",
                                                               "l_partkey"
                                                              ],
                                          is_restricting = {"p_partkey": ["p_name", "p_mfgr", "p_brand",
                                                                          "p_type", "p_size", "p_container",
                                                                          "p_retailprice",  "p_comment",
                                                                          "l_partkey", "ps_partkey"]
                                                     },
                                              row_count = 200000)
        self.matcher["partsupp"] = self.Tables("partsupp", 
                                               "ps_",
                                               ["ps_partkey", "ps_suppkey",
                                                "ps_availqty", "ps_supplycost",
                                                "ps_comment"],
                                               ["ps_partkey", "ps_suppkey"],
                                               ["PK__PARTSUPP__54937F6995C19898"],
                                              is_restricting = {"ps_partkey": ["p_partkey"],
                                                                "ps_suppkey": ["s_suppkey"]
                                                     },
                                              row_count = 800000)
        self.matcher["region"] = self.Tables("region", 
                                             "r_",
                                             ["r_regionkey",
                                              "r_name",
                                              "r_comment"],
                                             ["R_REGIONKEY"],
                                             ["PK__REGION__F403C3F000643726"],
                                             field_restrictions = ["n_regionkey",
                                                                   "c_nationkey",
                                                                   "r_name",
                                                                   "s_nationkey",
                                                                   "r_regionkey",
                                                                 ],
                                             is_restricting = {"r_regionkey": ["r_name", "r_comment", "n_regionkey"],
                                                               #"R_NAME": ["R_REGIONKEY", "R_COMMENT"]
                                                     },
                                              row_count = 5
                                            )
        self.matcher["supplier"] = self.Tables("supplier", 
                                               "s_",
                                               ["s_suppkey", "s_name", 
                                                "s_address", "s_nationkey",
                                                "s_phone", "s_acctbal",
                                                "s_comment"],
                                               ["s_suppkey"],
                                               ["PK__SUPPLIER__3632082B20AB4470"],
                                               field_restrictions = ["ps_suppkey",
                                                                     "s_suppkey",
                                                                     "l_suppkey"
                                                                   ],
                                               is_restricting = {"s_suppkey": ["s_name", "s_address",
                                                                               "s_nationkey", "s_phone", 
                                                                               "s_acctbal", "s_comment",
                                                                               "l_suppkey", "ps_suppkey"],
                                                                 "s_nationkey": ["n_nationkey"]
                                                                },
                                              row_count = 10000)
        self.prefix_matcher = self.prefix_to_table()
    
    def prefix_to_table(self):
        result = {}
        for key in self.matcher.keys():
            result[self.matcher[key].prefix.lower()] = self.matcher[key]
        return result
    
    def table_to_prefix(self, table_name):
        return self.matcher[table_name.lower()].prefix
    
    
    def match_prefix(self, pre):
        return self.prefix_matcher[pre.lower()]
    
    def get_table(self, table_name):
        return self.matcher[table_name.lower()]
    
    def match_from_column(self, column_name):
        prefix = (column_name.split("_")[0]+"_").lower()
        return self.match_prefix(prefix)
    
    def get_corresponding_restricted_fields(self, fields):
        result_fields = fields
        for key in self.matcher:
            if any([field in self.matcher[key].field_restrictions for field in fields]):
                result_fields.extend(self.matcher[key].field_restrictions)
        return result_fields
    
    def get_further_reduced_columns(self, fields, joins):
        result_reduced_columns = copy.deepcopy(fields)
        while len(fields):
            f = fields.pop(0)
            curr_table = self.match_from_column(f)
            if f in curr_table.is_restricting.keys():
                for res_field in curr_table.is_restricting[f]:
                    if not res_field.startswith(curr_table.prefix):
                        other_table = self.match_prefix(res_field.split("_")[0]+"_").table_name
                        if not any([res_field in join and f in join and
                                    ((curr_table.table_name == join[0] and other_table == join[2]) or
                                    (curr_table.table_name == join[2] and other_table == join[0])) and
                                    f in join for join in joins]):
                            continue
                    if res_field not in result_reduced_columns:
                        result_reduced_columns.append(res_field)
                        fields.append(res_field)
        return result_reduced_columns
    
    def has_join_connection(self, table_1, table_2, joins):
        for join in joins:
            if table_1 in join and table_2 in join:
                return True
        return False
    
    
    class Tables:
        # field restrictions are currently limited to the fields regarded in datafarm
        def __init__(self, table_name, prefix, columns, keys, key_name_server, field_restrictions=[], is_restricting = {}, row_count = 0):
            """
            :param field_restrictions: fields which reduce the table to one row if there is an equality-filter on this field
            """
            self.table_name = table_name
            self.prefix = prefix
            self.columns = columns
            self.keys = keys
            self.key_name_server = key_name_server
            self.field_restrictions = field_restrictions
            self.is_restricting = is_restricting
            self.row_count = row_count
            
        def get_columns(self):
            return self.columns
        
        def get_first_key(self):
            if self.has_keys():
                return self.keys[0]
            return None
        
        def get_keys(self):
            if self.has_keys():
                return self.keys
            return []
        
        def has_keys(self):
            return len(self.keys) > 0
            
        def get_server_key(self):
            return self.key_name_server[0]
        
        def has_column(self, column_name):
            for col in self.columns:
                if col.endswith(column_name):
                    return col
            return ""
        
        def get_prefix(self):
            return self.prefix