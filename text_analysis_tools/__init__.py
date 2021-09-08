# -*- coding: utf-8 -*-

from text_analysis_tools.api.text_cluster.kmeans import KmeansClustering
from text_analysis_tools.api.text_cluster.dbscan import DbscanClustering
from text_analysis_tools.api.text_similarity.cosion import CosionSimilarity
from text_analysis_tools.api.text_similarity.edit import EditSimilarity
from text_analysis_tools.api.text_similarity.simhash import SimHashSimilarity
from text_analysis_tools.api.keywords.tfidf import TfidfKeywords
from text_analysis_tools.api.keywords.textrank import TextRankKeywords
from text_analysis_tools.api.keyphrase.keyphrase import KeyPhraseExtraction
from text_analysis_tools.api.sentiment.sentiment import SentimentAnalysis
from text_analysis_tools.api.spell_correct.spell_correct import SpellCorrect
from text_analysis_tools.api.summarization.tfidf_summarization import TfidfSummarization
from text_analysis_tools.api.summarization.textrank_summarization import TextRankSummarization
from text_analysis_tools.api.topic_keywords.topic_kwywords import TopicKeywords
from text_analysis_tools.api.text_classification.fasttext import Fasttext
from text_analysis_tools.api.synonym.word2vec import Word2VecSynonym
from text_analysis_tools.api.synonym.synonym_dict import SynonymDict
from text_analysis_tools.api.triple_extraction.triple_extraction_ltp import TripleExtraction