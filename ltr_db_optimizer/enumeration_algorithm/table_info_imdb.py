import copy

class IMDBTableInformation:
    schema = "dbo"
    database = "imdb" 
    
    def __init__(self, alias_dict = {"Tables": [], "Filters": []}):
        self.matcher = {}
        self.alias_dict = alias_dict
        self.matcher["aka_name"] = self.Tables("aka_name",
                                               "",
                                               ["id", "person_id",
                                                "name", "imdb_index",
                                                "name_pcode_cf", "name_pcode_nf",
                                                "surname_pcode", "md5sum"],
                                               ["id"],
                                               ["PK_aka_name"], 
                                               is_restricting = {"aka_name.id": ["aka_name.person_id", "aka_name.name", "aka_name.imdb_index",
                                                                        "aka_name.name_pcode_cf", "aka_name.name_pcode_nf",
                                                                        "aka_name.surname_pcode", "aka_name.md5sum"],},
                                              row_count = 901343)
        self.matcher["aka_title"] = self.Tables("aka_title",
                                               "",
                                               ["id", "movie_id", "title",
                                                "imdb_index", "kind_id", "production_year",
                                                "phonetic_code", "episode_of_id", "season_nr",
                                                "episode_nr", "note", "md5sum"],
                                               ["id"],
                                               ["PK_aka_title"], 
                                               is_restricting = {"aka_title.id": ["aka_title.movie_id", "aka_title.title",
                                                                        "aka_title.imdb_index", "aka_title.kind_id", "aka_title.production_year",
                                                                        "aka_title.phonetic_code", "aka_title.episode_of_id", "aka_title.season_nr",
                                                                        "aka_title.episode_nr", "aka_title.note", "aka_title.md5sum"],},
                                              row_count = 316472)
        self.matcher["cast_info"] = self.Tables("cast_info",
                                               "",
                                               ["id", "person_id", "movie_id",
                                                "person_role_id", "note", "nr_order",
                                                "role_id"],
                                               ["id"],
                                               ["PK_cast_info"], 
                                               is_restricting = {"cast_info.id": ["cast_info.person_id", "cast_info.movie_id",
                                                                        "cast_info.person_role_id", "cast_info.note", "cast_info.nr_order",
                                                                        "cast_info.role_id"],},
                                              row_count = 36244343)       
        self.matcher["char_name"] = self.Tables("char_name",
                                               "",
                                               ["id", "name", "imdb_index", "imdb_id",
                                                "name_pcode_nf", "surname_pcode", "md5sum"],
                                               ["id"],
                                               ["PK_char_name"], 
                                               is_restricting = {"char_name.id": ["char_name.name", "char_name.imdb_index", "char_name.imdb_id",
                                                                        "char_name.name_pcode_nf", "char_name.surname_pcode", "char_name.md5sum"],},
                                              row_count = 3140339)
        self.matcher["comp_cast_type"] = self.Tables("comp_cast_type",
                                                     "",
                                                     ["id", "kind"],
                                                     ["id"],
                                                     ["PK_comp_cast_type"], 
                                                     is_restricting = {"comp_cast_type.id": ["comp_cast_type.kind"],},
                                                     row_count = 4)
        self.matcher["company_name"] = self.Tables("company_name",
                                                   "",
                                                   ["id", "name", "country_code",
                                                    "imdb_id", "name_pcode_nf",
                                                    "name_pcode_sf", "md5sum"],
                                                   ["id"],
                                                   ["PK_company_name"], 
                                                   is_restricting = {"company_name.id": ["company_name.name", "company_name.country_code",
                                                                            "company_name.imdb_id", "company_name.name_pcode_nf",
                                                                            "company_name.name_pcode_sf", "company_name.md5sum"]},
                                                   row_count = 234997)
        self.matcher["company_type"] = self.Tables("company_type",
                                                   "",
                                                   ["id", "kind"],
                                                   ["id"],
                                                   ["PK_company_type"], 
                                                   is_restricting = {"company_type.id": ["company_type.kind"],
                                                                     "company_type.kind": ["company_type.id"]},
                                                   row_count = 4)
        self.matcher["complete_cast"] = self.Tables("complete_cast",
                                                   "",
                                                   ["id", "movie_id", "subject_id",
                                                    "status_id"],
                                                   ["id"],
                                                   ["PK_complete_cast"], 
                                                   is_restricting = {"complete_cast.id": ["complete_cast.movie_id", "complete_cast.subject_id",
                                                                            "complete_cast.status_id"]},
                                                   row_count = 135086)
        self.matcher["info_type"] = self.Tables("info_type",
                                                   "",
                                                   ["id", "info"],
                                                   ["id"],
                                                   ["PK_info_type"], 
                                                   is_restricting = {"info_type.id": ["info_type.info"],
                                                                     "info_type.info": ["info_type.id"],
                                                                    },
                                                   row_count = 113)
        self.matcher["keyword"] = self.Tables("keyword",
                                              "",
                                              ["id", "keyword", "phonetic_code"],
                                              ["id"],
                                              ["PK_keyword"], 
                                              is_restricting = {"keyword.id": ["keyword.keyword", "keyword.phonetic_code"],
                                                                "keyword.keyword": ["keyword.id", "keyword.phonetic_code"],
                                                               },
                                              row_count = 134170)
        self.matcher["kind_type"] = self.Tables("kind_type",
                                                "",
                                                ["id", "kind"],
                                                ["id"],
                                                ["PK_kind_type"], 
                                                is_restricting = {"kind_type.id": ["kind_type.kind"]},
                                                row_count = 7)
        self.matcher["link_type"] = self.Tables("link_type",
                                                "",
                                                ["id", "link"],
                                                ["id"],
                                                ["PK_link_type"], 
                                                is_restricting = {"link_type.id": ["link_type.link"]},
                                                row_count = 18)
        self.matcher["movie_companies"] = self.Tables("movie_companies",
                                               "",
                                               ["id", "movie_id", "company_id",
                                                "company_type", "note"],
                                               ["id"],
                                               ["PK_movie_companies"], 
                                               is_restricting = {"movie_companies.id": ["movie_companies.movie_id", "movie_companies.company_id",
                                                                        "movie_companies.company_type", "movie_companies.note"],},
                                              row_count = 2609129)
        self.matcher["movie_info"] = self.Tables("movie_info",
                                               "",
                                               ["id", "movie_id", "info_type_id",
                                                "info", "note"],
                                               ["id"],
                                               ["PK_movie_info"], 
                                               is_restricting = {"movie_info.id": ["movie_info.movie_id", "movie_info.info_type_id",
                                                                                   "movie_info.info", "movie_info.note"],},
                                              row_count = 1380035)
        self.matcher["movie_info_idx"] = self.Tables("movie_info_idx",
                                               "",
                                               ["id", "movie_id", "info_type_id",
                                                "info", "note"],
                                               ["id"],
                                               ["PK_movie_info_idx"], 
                                               is_restricting = {"movie_info_idx.id": ["movie_info_idx.movie_id", "movie_info_idx.info_type_id",
                                                                        "movie_info_idx.info", "movie_info_idx.note"],},
                                              row_count = 1380035)
        self.matcher["movie_keyword"] = self.Tables("movie_keyword",
                                                    "",
                                                    ["id", "movie_id", "keyword_id"],
                                                    ["id"],
                                                    ["PK_movie_keyword"], 
                                                    is_restricting = {"movie_keyword.id": ["movie_keyword.movie_id", "movie_keyword.keyword_id"]},
                                              row_count = 4523930)
        self.matcher["movie_link"] = self.Tables("movie_link",
                                                    "",
                                                    ["id", "movie_id", "linked_movie_id", "link_type_id"],
                                                    ["id"],
                                                    ["PK_movie_link"], 
                                                    is_restricting = {"movie_link.id": ["movie_link.movie_id", "movie_link.linked_movie_id",
                                                                             "movie_link.link_type_id"]},
                                              row_count = 29997)
        self.matcher["name"] = self.Tables("name",
                                               "",
                                               ["id", "name", "imdb_index", "imdb_id",
                                                "gender", "name_pcode_cf", "name_pcode_nf",
                                                "surname_pcode", "md5sum"],
                                               ["id"],
                                               ["PK_name"], 
                                               is_restricting = {"name.id": ["name.name", "name.imdb_index", "name.imdb_id",
                                                                        "name.gender", "name.name_pcode_cf", "name.name_pcode_nf",
                                                                        "name.surname_pcode", "name.md5sum"],},
                                              row_count = 4167491)
        self.matcher["person_info"] = self.Tables("person_info",
                                               "",
                                               ["id", "person_id", "info_type_id", 
                                                "info", "note"],
                                               ["id"],
                                               ["PK_person_info"], 
                                               is_restricting = {"person_info.id": ["person_info.person_id", "person_info.info_type_id", 
                                                                        "person_info.info", "person_info.note"],},
                                              row_count = 2963652)
        self.matcher["role_type"] = self.Tables("role_type",
                                                "",
                                                ["id", "role"],
                                                ["id"],
                                                ["PK_role_type"], 
                                                is_restricting = {"role_type.id": ["role_type.role"],
                                                                  "role_type.role": ["role_type.id"]},
                                                row_count = 12)
        self.matcher["title"] = self.Tables("title",
                                               "",
                                               ["id", "title", "imdb_index", "kind_id", "production_year",
                                                "imdb_id", "phonetic_code", "episode_of_id", "season_nr",
                                                "episode_nr", "series_years", "md5sum"],
                                               ["id"],
                                               ["PK_title"], 
                                               is_restricting = {"title.id": ["title.title", "title.imdb_index", "title.kind_id", "title.production_year",
                                                                        "title.imdb_id", "title.phonetic_code", "title.episode_of_id", "title.season_nr",
                                                                        "title.episode_nr", "title.series_years", "title.md5sum"],},
                                              row_count = 2528312)

    
    def get_table(self, table_name):
        if table_name in self.alias_dict["Tables"]:
            table_name = self.alias_dict["Tables"][table_name]
        return self.matcher[table_name.lower()]
    
    def get_corresponding_restricted_fields(self, fields):
        result_fields = fields
        for key in self.matcher:
            if any([field in self.matcher[key].field_restrictions for field in fields]):
                result_fields.extend(self.matcher[key].field_restrictions)
        return result_fields
    
    def get_further_reduced_columns(self, fields, joins):
        result_reduced_columns = []
        while len(fields):
            f = fields.pop(0)
            if "." not in f:
                continue
            f_splitted = f.split(".")
            if f_splitted[0] in self.alias_dict["Tables"]: 
                curr_table = self.alias_dict["Tables"][f_splitted[0]]
                f = f.replace(f_splitted[0],curr_table,1)
                curr_table = self.get_table(curr_table)
            else:
                curr_table = self.get_table(f_splitted[0])
            if f not in result_reduced_columns:
                result_reduced_columns.append(f)
            if f in curr_table.is_restricting:
                for res_field in curr_table.is_restricting[f]:
                    #if not res_field.startswith(curr_table.prefix):
                    #    other_table = self.match_prefix(res_field.split("_")[0]+"_").table_name
                    #    if not any([res_field in join and f in join and
                    #                ((curr_table.table_name == join[0] and other_table == join[2]) or
                    #                (curr_table.table_name == join[2] and other_table == join[0])) and
                    #                f in join for join in joins]):
                    #        continue
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