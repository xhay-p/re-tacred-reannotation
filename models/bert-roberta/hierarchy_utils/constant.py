"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

NEGATIVE_LABEL = 'no_relation'

LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}
LABEL_TO_ID_HIER = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41, 'per:nationality': 42, 'org:location_of_headquarters':43, 'per:location_of_birth':44, 'per:location_of_death':45, 'per:locations_of_residence':46, 'per:family':47, 'per-per':48, 'per-org':49, 'per-misc':50, 'per-loc':51, 'org-per':52, 'org-org':53, 'org-misc':54, 'org-loc':55, 'per':56, 'org':57, 'relation':58, 'root':59}
BINARY_LABEL_TO_ID = {'no_relation': 0, 'relation': 1}
BINARY_MAP = {'per:title': 'relation', 'org:top_members/employees': 'relation', 'per:employee_of': 'relation',
              'org:alternate_names': 'relation', 'org:country_of_headquarters': 'relation',
              'per:countries_of_residence': 'relation', 'org:city_of_headquarters': 'relation',
              'per:cities_of_residence': 'relation', 'per:age': 'relation',
              'per:stateorprovinces_of_residence': 'relation', 'per:origin': 'relation', 'org:subsidiaries': 'relation',
              'org:parents': 'relation', 'per:spouse': 'relation', 'org:stateorprovince_of_headquarters': 'relation',
              'per:children': 'relation', 'per:other_family': 'relation', 'per:alternate_names': 'relation',
              'org:members': 'relation', 'per:siblings': 'relation', 'per:schools_attended': 'relation',
              'per:parents': 'relation', 'per:date_of_death': 'relation', 'org:member_of': 'relation',
              'org:founded_by': 'relation', 'org:website': 'relation', 'per:cause_of_death': 'relation',
              'org:political/religious_affiliation': 'relation', 'org:founded': 'relation',
              'per:city_of_death': 'relation', 'org:shareholders': 'relation',
              'org:number_of_employees/members': 'relation', 'per:date_of_birth': 'relation',
              'per:city_of_birth': 'relation', 'per:charges': 'relation', 'per:stateorprovince_of_death': 'relation',
              'per:religion': 'relation', 'per:stateorprovince_of_birth': 'relation',
              'per:country_of_birth': 'relation', 'org:dissolved': 'relation', 'per:country_of_death': 'relation'}

PARENTS = {'org:city_of_headquarters': 'org:location_of_headquarters',
          'org:country_of_headquarters': 'org:location_of_headquarters',
          'org:stateorprovince_of_headquarters': 'org:location_of_headquarters',
          'per:city_of_birth': 'per:location_of_birth',
          'per:country_of_birth': 'per:location_of_birth',
          'per:stateorprovince_of_birth': 'per:location_of_birth',
          'per:city_of_death': 'per:location_of_death',
          'per:country_of_death': 'per:location_of_death',
          'per:stateorprovince_of_death': 'per:location_of_death',
          'per:cities_of_residence' : 'per:locations_of_residence',  
          'per:countries_of_residence' : 'per:locations_of_residence',
          'per:stateorprovinces_of_residence' : 'per:locations_of_residence',
          'per:children' : 'per:family',
          'per:other_family' : 'per:family',
          'per:parents' : 'per:family',
          'per:siblings' : 'per:family',
          'per:spouse' : 'per:family'}
RELABELED_LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 
            'org:location_of_headquarters': 5, 'per:locations_of_residence': 6, 'per:age': 7, 'per:origin': 8, 'org:subsidiaries': 9,
            'org:parents': 10, 'per:family': 11, 'per:alternate_names': 12, 'org:members': 13, 'per:schools_attended': 14, 
            'per:date_of_death': 15, 'org:member_of': 16, 'org:founded_by': 17, 'org:website': 18, 'per:cause_of_death': 19, 
            'org:political/religious_affiliation': 20, 'org:founded': 21, 'per:location_of_death': 22, 'org:shareholders': 23, 
            'org:number_of_employees/members': 24, 'per:date_of_birth': 25, 'per:location_of_birth': 26, 'per:charges': 27, 
            'per:religion': 28, 'org:dissolved': 29}
ALLPOS_LABEL_TO_ID = {'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

FILTER_OUT = [('ORGANIZATION', 'org:alternate_names', 'MISC'),
              ('ORGANIZATION', 'org:member_of', 'COUNTRY'),
              ('ORGANIZATION', 'org:member_of', 'LOCATION'),
              ('ORGANIZATION', 'org:member_of', 'STATE_OR_PROVINCE'),
              ('ORGANIZATION', 'org:parents', 'COUNTRY'),
              ('ORGANIZATION', 'org:parents', 'LOCATION'),
              ('ORGANIZATION', 'org:parents', 'STATE_OR_PROVINCE'),
              ('ORGANIZATION', 'org:subsidiaries', 'COUNTRY'),
              ('ORGANIZATION', 'org:subsidiaries', 'LOCATION'),
              ('PERSON', 'per:alternate_names', 'MISC'),
              ('PERSON', 'per:employee_of', 'LOCATION'),
             ]

INFINITY_NUMBER = 1e12
