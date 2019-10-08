# -*- coding: utf-8 -*-

import json
from text_analysis_tools.api.synonym import SYNONYM_DICT


class SynonymDict:
    def __init__(self):
        self.synonym_dict = self.load_json()

    def load_json(self):
        with open(SYNONYM_DICT, encoding='utf-8') as f:
            return json.loads(f.read())

    def synonym(self, word):
        try:
            ret = self.synonym_dict[word]
        except:
            ret = []
        return ret
