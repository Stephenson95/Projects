# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:32:56 2026

@author: Stephenson
"""
from rapidfuzz import fuzz, distance

string1 = "30 Imails Lane Don Valley VIC 3139"
string2 = "30 Ismail Lane Launching Place VIC 3139"

fuzz.ratio(string1, string2)
fuzz.WRatio(string1, string2)
fuzz.QRatio(string1, string2)
fuzz.partial_ratio(string1, string2)
fuzz.token_set_ratio(string1, string2)
fuzz.token_sort_ratio(string1, string2)


distance.Levenshtein.normalized_distance(string1, string2)
distance.Levenshtein.normalized_similarity(string1, string2, weights=(1,1,2))
distance.JaroWinkler.distance(string1, string2) #0 is good
