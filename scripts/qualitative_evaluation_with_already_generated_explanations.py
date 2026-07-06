#!/usr/bin/env python3
# coding=utf-8
# Author: Alberto Olivares Alarcos <aolivares@iri.upc.edu>, Institut de Robòtica i Informàtica Industrial, CSIC-UPC
"""
code to evaluate already generated explanations from ontology-based narratives using language models 
(in this code, an agent from pydantic-ai is used)
"""

from src.utils_module import utilsModule, testUtilsModule
from src.pydanticai_module import *
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

import time
import textstat
import pandas as pd
import numpy as np




if __name__ == "__main__":
    utils_object = utilsModule()
    test_utils_object = testUtilsModule()

    model_used_to_generate_explanations = 'gpt-oss:20b' # qwen3:14b  |  qwen3:8b  |  qwen3:1.7b  |  qwen3:32b  |  gpt-oss:20b  
    model_used_to_evaluate_explanations = 'qwen3.5:9b' # qwen3.5:9b  |  gemma4:12b / 26b  |  gpt-oss:20b
    specificity = 3
    dataset_name = 'collaborative_ssd_case_inspection'

    test_utils_object.init_semantic_similarity_metric_with_llm_as_judge(model_used_to_evaluate_explanations)

    # Pre-trained encoder models
    bi_encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder_model = CrossEncoder("cross-encoder/stsb-distilroberta-base") # great for Semantic Textual Similarity (only English)
    #cross_encoder_model = CrossEncoder("cross-encoder/distiluse-base-multilingual-cased-v1") # useful in multilingual comparisons

    evaluation_results_file = "qualitative_evaluation_results_article_example_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" \
        + model_used_to_generate_explanations + "_for_generation_" \
        + model_used_to_evaluate_explanations + "_for_evaluation" + ".csv"


    results_dict = {"Pair ID": list(), "Dataset" : list(), "Language model" : list(), "Specificity" : list(), \
                     "Words": list(), "Readability score": list(), "Cosine similarity": list(), \
                     "Bi-encoder similarity": list(), "Cross-encoder similarity": list(), "LLM-judge similarity": list(), \
                     "LLM-judge similarity SD": list(), "LLM-judge similarity MAD": list()}

    explanations_dict = {"ID": list(), "Explanation" : list()}

    narrative = "‘Plan X’ is cheaper plan than and is shorter plan than and is faster plan than and is better plan than ‘Plan Y’. " +\
                "‘Plan X’ has makespan ‘Plan X makespan’; while ‘Plan Y’ has makespan ‘Plan Y makespan’. " +\
                "‘Plan X’ has number of tasks ‘Plan X number of tasks’; while ‘Plan Y’ has number of tasks ‘Plan Y number of tasks’. " +\
                "‘Plan X’ has cost ‘Plan X cost’; while ‘Plan Y’ has cost ‘Plan Y cost’. ‘Plan X’ is classified by ‘TypicalPlan’; " +\
                "while ‘Plan Y’ is classified by ‘AtypicalPlan’. ‘Plan X makespan’ has value ‘11.20’; " +\
                "while ‘Plan Y makespan’ has value ‘22.35’. ‘Plan X number of tasks’ has value ‘13’; " +\
                "while ‘Plan Y number of tasks’ has value ‘17’. ‘Plan X cost’ has value ‘1’; while ‘Plan Y cost’ has value ‘5’."              
    
    explanations_dict["ID"].append("correct explanation")
    explanations_dict["Explanation"].append("Plan X is cheaper, faster, shorter, and better than Plan Y. Plan X takes 11.20 time units, " +\
                                       "has 13 tasks, and costs 1. Plan Y takes 22.35 time units, has 17 tasks, and costs 5. " +\
                                       "Plan X is a typical plan; Plan Y is an atypical plan.")

    explanations_dict["ID"].append("minor numeric error")
    explanations_dict["Explanation"].append("Plan X is cheaper, faster, shorter, and better than Plan Y. Plan X takes 11.20 time units, " +\
                                       "has 13 tasks, and costs 1. Plan Y takes 22 time units, has 17 tasks, and costs 5. " +\
                                       "Plan X is a typical plan; Plan Y is an atypical plan.")
                        
    explanations_dict["ID"].append("major numeric error")
    explanations_dict["Explanation"].append("Plan X is cheaper, faster, shorter, and better than Plan Y. Plan X takes 11.20 time units, " +\
                                       "has 13 tasks, and costs 1. Plan Y takes 12 time units, has 17 tasks, and costs 5. " +\
                                       "Plan X is a typical plan; Plan Y is an atypical plan.")

    explanations_dict["ID"].append("semantic error i - inversed typicality")
    explanations_dict["Explanation"].append("Plan X is cheaper, faster, shorter, and better than Plan Y . Plan X takes 11.20 time units, " +\
                                       "has 13 tasks, and costs 1. Plan Y takes 22.35 time units, has 17 tasks, and costs 5. " +\
                                       "Plan X is an atypical plan; Plan Y is a typical plan.")

    explanations_dict["ID"].append("semantic error ii - inversed better plan")
    explanations_dict["Explanation"].append("Plan X is cheaper, faster, shorter, and worse than Plan Y. Plan X takes 11.20 time units, " +\
                                       "has 13 tasks, and costs 1. Plan Y takes 22.35 time units, has 17 tasks, and costs 5. " +\
                                       "Plan X is a typical plan; Plan Y is an atypical plan.")
    
    explanations_dict["ID"].append("semantic error iii - inversed shorter plan")
    explanations_dict["Explanation"].append("Plan X is cheaper, faster, larger, and better than Plan Y. Plan X takes 11.20 time units, " +\
                                       "has 13 tasks, and costs 1. Plan Y takes 22.35 time units, has 17 tasks, and costs 5. " +\
                                       "Plan X is a typical plan; Plan Y is an atypical plan.")

    number_of_explanations = len(explanations_dict["Explanation"])
    for i in range (0, number_of_explanations):    
        results_dict['Pair ID'].append(explanations_dict['ID'][i])
        results_dict['Dataset'].append(dataset_name)
        results_dict['Language model'].append(model_used_to_generate_explanations)
        results_dict['Specificity'].append(specificity)
        results_dict['Words'].append(len(explanations_dict['Explanation'][i].split()))
        ##results_dict['Readability score'].append(textstat.dale_chall_readability_score(result.output.narrative))
        ##results_dict['Readability score'].append(textstat.text_standard(result.output.narrative, float_output=True))
        results_dict['Readability score'].append(textstat.flesch_reading_ease(explanations_dict['Explanation'][i]))
        results_dict['Cosine similarity'].append(test_utils_object.compute_text_cosine_similarity(\
            [narrative, explanations_dict['Explanation'][i]])) # item() turns the np.float into python float
        results_dict['Bi-encoder similarity'].append(bi_encoder_model.similarity(\
            bi_encoder_model.encode(narrative), \
            bi_encoder_model.encode(explanations_dict['Explanation'][i])).item()) # cosine-similarity by default (item is used to extract the value of the result, which is a tensor)
        results_dict['Cross-encoder similarity'].append(cross_encoder_model.predict([narrative, explanations_dict['Explanation'][i]]))
        
        # the next metric is non-deterministic thus we evaluate its consistency
        llm_judge_score_list = list()
        for j in range (0, 100):
            llm_judge_score_list.append(test_utils_object.compute_semantic_similarity_with_llm_as_judge(\
                narrative, explanations_dict['Explanation'][i])['score'])
        median = np.median(llm_judge_score_list)
        sd = np.std(llm_judge_score_list)
        mad = np.median(np.abs(llm_judge_score_list - median))
        
        """
        Low MAD + Low SD (Ideal Consistency): The LLM returns the exact same score (or very close) every single time. It is highly deterministic and stable.
        Low MAD + High SD (The "Hallucinator"): The LLM gives the identical score 95% of the time, but occasionally throws out a wildly different, erratic score due to temperature or formatting shifts. MAD correctly tracks the tight consensus, while SD flags the dangerous outliers.
        High MAD + High SD (The "Flip-Flopper"): The LLM is highly sensitive to randomness and fluctuates wildly across the entire scoring spectrum on every run.
        """
        results_dict['LLM-judge similarity'].append(np.average(llm_judge_score_list))
        results_dict['LLM-judge similarity SD'].append(sd) # Standard Deviation (measures outliers sensitivity)
        results_dict['LLM-judge similarity MAD'].append(mad) # Median Absolute Deviation (measures consistency)
        
        #results_dict['LLM-judge similarity'].append(test_utils_object.compute_semantic_similarity_with_llm_as_judge(\
        #    narrative, explanations_dict['Explanation'][i])['score'])

        print('Execution number: ', i)
    

    df = pd.DataFrame.from_dict(results_dict, orient='columns')
    df.to_csv(utils_object.csv_file_path + "/" + evaluation_results_file, index=False) # data frame

    print("\n ·· TEST ·····\n Explanation number of words (aprox.): ", results_dict["Words"]) 
    print(" Explanation readability (score): ", results_dict["Readability score"]) 
    print(" Explanation cosine similary: ", results_dict["Cosine similarity"]) 
    print(" Explanation bi-encoder similary: ", results_dict["Bi-encoder similarity"]) 
    print(" Explanation cross-encoder similary: ", results_dict["Cross-encoder similarity"]) 
    print(" Explanation llm-judge similarity: ", results_dict["LLM-judge similarity"]) 
    print(" Explanation llm-judge similarity SD: ", results_dict["LLM-judge similarity SD"]) 
    print(" Explanation llm-judge similarity MAD: ", results_dict["LLM-judge similarity MAD"]) 
    print("\n") 