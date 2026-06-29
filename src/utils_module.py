#!/usr/bin/env python3
# coding=utf-8
# Author: Alberto Olivares Alarcos <aolivares@iri.upc.edu>, Institut de Robòtica i Informàtica Industrial, CSIC-UPC


import os
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import OllamaModel

class utilsModule:
    def __init__(self):
        aux_ = os.path.abspath(__file__)
        self.repository_name = "know_rox"
        self.main_path = aux_[:aux_.find(self.repository_name)+len(self.repository_name)] # find starting pose and add upto that pose + length
        self.csv_file_path = self.main_path + "/csv"

    def create_dict_from_csv_file(self, csv_file_path, csv_file_name, separator):
        """
        Creates a dictionary from the data stored in a csv file.

        Args:
            csv_file_path: a path to a csv file (without the last '/', it is added here)
            csv_file_name: a csv file name (including the .csv)
            separator: the separator used in the file

        Returns:
            A dictionary with the first row of the csv file as keys and the rest of the values 
            as a list (for each key).
        """

        out_dict = dict()

        data = pd.read_csv(csv_file_path + "/" + csv_file_name, sep=separator) #data frame
        out_dict = data.to_dict(orient='list') # dict of lists (first row as 'k')
        #print(out_dict)
        
        return out_dict
    
    def create_csv_file_from_dict(self, dict_in, csv_file_path, csv_file_name):
        """
        Creates a dictionary from the data stored in a csv file.

        Args:
            dict_in: a dictionary to be stored as CSV file
            csv_file_path: a path to a csv file (without the last '/', it is added here)
            csv_file_name: a csv file name (including the .csv)

        Returns:
            None.
        """

        out_dict = dict()

        df = pd.DataFrame.from_dict(dict_in, orient='columns')
        df.to_csv(csv_file_path + '/' + csv_file_name, index=False) # data frame    




class testUtilsModule:
    def __init__(self):
        # Variables
        self.memory_usage_ = 0

        self.evaluation_model_to_run = 'gpt-oss:20b' # qwen3:14b  |  qwen3:8b  |  qwen3:1.7b  |  qwen3:32b  |  gpt-oss:20b  
        self.ollama_judge = OllamaModel(
            model=self.evaluation_model_to_run, 
            base_url="http://localhost:11434",
            temperature=0.0
        ) # initialize the local Ollama judge model with 0 temperature for more deterministic grading behavior.

        # define a Custom Semantic Similarity Metric using G-Eval
        self.semantic_similarity_metric = GEval(
            name="Semantic Coherence Similarity",
            model=self.ollama_judge,
            # Define the parameters G-Eval needs to look at from the test case
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            # The exact criteria the judge will grade against
            evaluation_steps=[
                "Compare the core semantic meaning of the actual output against the input narrative.",
                "Ignore changes in tone, formatting, and minor stylistic embellishments that make the text sound more 'natural'.",
                "Penalize heavily if semantic facts, core semantic relationship dynamics, or (numeric) data are missing or altered.",
                "Determine if a human reading the actual output would walk away with the exact same understanding as reading the input narrative.",
                "Rate the similarity on a scale from 0 (completely different meaning) to 1 (perfect semantic equivalence)."
            ],
            threshold=0.75  # Pass/Fail threshold for your pipeline tests
        )

    def get_memory_usage(self): 
        """
        Parses /proc/self/status to extract relevant memory figures. [ONLY tested wit Linux, it does not work on macOS]

        Args: 
            None
    
        Returns:
            memuse : a dict from str (name of memory figure) to Float (size in MB)
        """
        memuse = {}
        with open("/proc/self/status") as status:
            # This is not very portable, only works in Unix-like systems      (like Linux).
            for line in status:
                parts = line.split()
                if parts[0].startswith("Vm"):
                    key = parts[0][:-1].lower()
                    memuse[key] = float(parts[1])/1024
        return memuse
    
    def compute_text_cosine_similarity(self, documents):
        """
        Computes the cosine similary between different documents.

        Args:
            documents: a list of documents (list(str))

        Returns:
            cosine_similarity_out: a matrix (or a float when only two documents) containing the similarity
        """
        index_matrix = list()
        count = 0
        for d in documents:
            index_matrix.append("doc_"+str(count))
            count += 1

        count_vectorizer = CountVectorizer(stop_words="english")
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)
        
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(
        doc_term_matrix,
        columns=count_vectorizer.get_feature_names_out(),
        index=index_matrix,
        )
        
        #print(df) # matrix of word frequency in all documents
        #print(cosine_similarity(df, df)) # matrix of len(documents) x len(documents) (the diagonal is 1)

        if len(documents) == 2: 
            cosine_similarity_out = cosine_similarity(df, df)[0][1]
        else:
            cosine_similarity_out = cosine_similarity(df, df)

        return cosine_similarity_out

    def compute_semantic_similarity_with_llm_as_judge(self, ground_truth: str, natural_explanation: str):
        """
        Computes the semantic similarity between the original ontology-based narrative 
        and the naturalized explanation.

        Args:
            ground_truth: a string containing the initial narrative
            natural_explanation: a string containing the final more natural explanation

        Returns:
            output_dictionary: a dictionary containing the similarity score, reason, and 
                               whether the score is higher a threshold 
        """
        # In DeepEval's LLMTestCase schema:
        # - `input` acts as our baseline anchor (Ground Truth Narrative)
        # - `actual_output` acts as what we are evaluating (Natural Explanation)
        test_case = LLMTestCase(
            input=ground_truth,
            actual_output=natural_explanation
        )
        
        # Run the metric evaluation
        self.semantic_similarity_metric.measure(test_case)
        
        # Return structured results
        return {
            "score": self.semantic_similarity_metric.score,
            "reason": self.semantic_similarity_metric.reason,
            "passed": self.semantic_similarity_metric.is_successful()
        }