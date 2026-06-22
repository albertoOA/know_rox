#!/usr/bin/env python3
# coding=utf-8
# Author: Alberto Olivares Alarcos <aolivares@iri.upc.edu>, Institut de Robòtica i Informàtica Industrial, CSIC-UPC
"""
code to evaluate already generated explanations from ontology-based narratives using language models 
(in this code, an agent from pydantic-ai is used)
"""

from src.utils_module import utilsModule, testUtilsModule
from src.pydanticai_module import *
from sentence_transformers.cross_encoder import CrossEncoder

import time
import textstat
import pandas as pd




if __name__ == "__main__":
    utils_object = utilsModule()
    test_utils_object = testUtilsModule()

    model_to_run = 'gpt-oss:20b' # qwen3:14b  |  qwen3:8b  |  qwen3:1.7b  |  qwen3:32b  |  gpt-oss:20b  
    specificity = 3
    dataset_name = 'collaborative_ssd_case_inspection'

    # Pre-trained cross encoder
    cross_encoder_model = CrossEncoder("cross-encoder/stsb-distilroberta-base") # great for Semantic Textual Similarity (only English)
    #cross_encoder_model = CrossEncoder("cross-encoder/distiluse-base-multilingual-cased-v1") # useful in multilingual comparisons

    narratives_csv_file = "generated_c_narratives_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + ".csv"
    explanations_csv_file = "generated_c_narratives_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" + model_to_run + ".csv"
    evaluation_results_file = "evaluation_results_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" + model_to_run + ".csv"

    narratives_dict = utils_object.create_dict_from_csv_file(\
        utils_object.csv_file_path + "/explanatory_narratives_cra/acxon_based", \
        narratives_csv_file, ",")

    explanations_dict = utils_object.create_dict_from_csv_file(\
        utils_object.csv_file_path + "/explanations_rox_based/" + dataset_name + "/" ,  \
        explanations_csv_file, ",")

    results_dict = {"Pair ID": list(), "Dataset" : list(), "Language model" : list(), "Specificity" : list(), \
                     "Words": list(), "Readability score": list(), "Cosine similarity": list(), \
                     "Cross-encoder similarity": list(), "LLM-judge similarity": list()}
    

    #number_of_explanations = 5
    number_of_explanations = len(narratives_dict["Explanation"])
    for i in range (0, number_of_explanations):    
        results_dict['Pair ID'].append(narratives_dict['Pair ID'][i])
        results_dict['Dataset'].append(dataset_name)
        results_dict['Language model'].append(model_to_run)
        results_dict['Specificity'].append(specificity)
        results_dict['Words'].append(len(explanations_dict['Explanation'][i].split()))
        ##results_dict['Readability score'].append(textstat.dale_chall_readability_score(result.output.narrative))
        ##results_dict['Readability score'].append(textstat.text_standard(result.output.narrative, float_output=True))
        results_dict['Readability score'].append(textstat.flesch_reading_ease(explanations_dict['Explanation'][i]))
        results_dict['Cosine similarity'].append(test_utils_object.compute_text_cosine_similarity(\
            [narratives_dict["Explanation"][i], explanations_dict['Explanation'][i]])) # item() turns the np.float into python float
        results_dict['Cross-encoder similarity'].append(cross_encoder_model.predict([narratives_dict["Explanation"][i], explanations_dict['Explanation'][i]]))
        results_dict['LLM-judge similarity'].append(test_utils_object.compute_semantic_similarity_with_llm_as_judge(\
            narratives_dict["Explanation"][i], explanations_dict['Explanation'][i])['score'])

        print('Execution number: ', i)
    

    df = pd.DataFrame.from_dict(results_dict, orient='columns')
    df.to_csv(utils_object.csv_file_path + "/" + evaluation_results_file, index=False) # data frame

    print("\n ·· TEST ·····\n Explanation number of words (aprox.): ", results_dict["Words"]) 
    print(" Explanation readability (score): ", results_dict["Readability score"]) 
    print(" Explanation cosine similary: ", results_dict["Cosine similarity"]) 
    print(" Explanation cross-encoder similary: ", results_dict["Cross-encoder similarity"]) 
    print(" Explanation llm-judge similary: ", results_dict["LLM-judge similarity"]) 
    print("\n") 