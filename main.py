"""
Author: LTH
Date: 2023-06-07 17:22:46
LastEditTime: 2023-06-07 17:22:50
FilePath: \da_chuang-2022-2023\main.py
Description: 
Copyright (c) 2023 by LTH, All Rights Reserved.
"""
from event_extraction import *
from event_library_construction import *
from event_contradiction_analysis import *

file_path = "input_article.txt"
article_name = file_path.replace(".txt", "")
output_path = f"information of {article_name}.txt"
