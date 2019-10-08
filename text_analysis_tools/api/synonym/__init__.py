# -*- coding: utf-8 -*-


import os

CURRENT_PATH = os.path.abspath(__file__)
DATA_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))
SYNONYM_DICT = os.path.join(DATA_PATH, "data", "synonym.json")