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

    model_used_to_generate_explanations = 'gpt-oss:20b' 
    specificity = 3
    dataset_name = 'collaborative_ssd_case_inspection'

    cross_encoder_model_for_natural_language_inferece_label_mapping = ["contradiction", "entailment", "neutral"]

    # Pre-trained encoder models
    bi_encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cross_encoder_model_for_text_similarity = CrossEncoder("cross-encoder/stsb-distilroberta-base") # great for Semantic Textual Similarity (only English)
    #cross_encoder_model_for_text_similarity = CrossEncoder("cross-encoder/distiluse-base-multilingual-cased-v1") # useful in multilingual comparisons
    cross_encoder_model_for_natural_language_inferece = CrossEncoder("cross-encoder/nli-deberta-v3-base")

    narratives_csv_file = "generated_c_narratives_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + ".csv"
    explanations_csv_file = "generated_c_narratives_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" + model_used_to_generate_explanations + ".csv"
    evaluation_results_file = "quantitative_evaluation_results_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" \
        + model_used_to_generate_explanations + "_for_generation_" + ".csv" 

    narratives_dict = utils_object.create_dict_from_csv_file(\
        utils_object.csv_file_path + "/explanatory_narratives_cra/acxon_based", \
        narratives_csv_file, ",")

    explanations_dict = utils_object.create_dict_from_csv_file(\
        utils_object.csv_file_path + "/explanations_rox_based/" + dataset_name + "/" ,  \
        explanations_csv_file, ",")

    results_dict = {"Pair ID": list(), "Dataset" : list(), "Language model" : list(), "Specificity" : list(), \
                     "Words": list(), "Readability score": list(), "Cosine similarity": list(), \
                     "Bi-encoder similarity": list(), "Cross-encoder similarity": list(), "Cross-encoder inference": list()}     

    #number_of_explanations = 10
    number_of_explanations = len(narratives_dict["Explanation"])
    for i in range (0, number_of_explanations):    
        results_dict['Pair ID'].append(narratives_dict['Pair ID'][i])
        results_dict['Dataset'].append(dataset_name)
        results_dict['Language model'].append(model_used_to_generate_explanations)
        results_dict['Specificity'].append(specificity)
        results_dict['Words'].append(len(explanations_dict['Explanation'][i].split()))
        ##results_dict['Readability score'].append(textstat.dale_chall_readability_score(result.output.narrative))
        ##results_dict['Readability score'].append(textstat.text_standard(result.output.narrative, float_output=True))
        results_dict['Readability score'].append(textstat.flesch_reading_ease(explanations_dict['Explanation'][i]))
        results_dict['Cosine similarity'].append(test_utils_object.compute_text_cosine_similarity(\
            [narratives_dict["Explanation"][i], explanations_dict['Explanation'][i]])) # item() turns the np.float into python float
        results_dict['Bi-encoder similarity'].append(bi_encoder_model.similarity(\
            bi_encoder_model.encode(narratives_dict["Explanation"][i]), \
            bi_encoder_model.encode(explanations_dict['Explanation'][i])).item()) # cosine-similarity by default (item is used to extract the value of the result, which is a tensor)
        results_dict['Cross-encoder similarity'].append(cross_encoder_model_for_text_similarity.predict([narratives_dict["Explanation"][i], explanations_dict['Explanation'][i]]))
        
        cross_encoder_inference_result_scores = cross_encoder_model_for_natural_language_inferece.predict([narratives_dict["Explanation"][i], explanations_dict['Explanation'][i]])
        cross_encoder_inference_result_label = cross_encoder_model_for_natural_language_inferece_label_mapping[np.argmax(cross_encoder_inference_result_scores)]
        results_dict['Cross-encoder inference'].append(cross_encoder_inference_result_label)


        print('Execution number: ', i)
    

    df = pd.DataFrame.from_dict(results_dict, orient='columns')
    df.to_csv(utils_object.csv_file_path + "/" + evaluation_results_file, index=False) # data frame

    print("\n ·· TEST ·····\n Explanation number of words (aprox.): ", results_dict["Words"]) 
    print(" Explanation readability (score): ", results_dict["Readability score"]) 
    print(" Explanation cosine similary: ", results_dict["Cosine similarity"]) 
    print(" Explanation bi-encoder similary: ", results_dict["Bi-encoder similarity"]) 
    print(" Explanation cross-encoder similary: ", results_dict["Cross-encoder similarity"]) 
    print(" Explanation cross-encoder inference: ", results_dict["Cross-encoder inference"]) 
    print("\n") 