#!/usr/bin/env python3
# coding=utf-8
# Author: Alberto Olivares Alarcos <aolivares@iri.upc.edu>, Institut de Robòtica i Informàtica Industrial, CSIC-UPC
"""
code to generate explanations from ontology-based narratives using language models 
(in this code, an agent from pydantic-ai is used)
"""

from src.utils_module import utilsModule, testUtilsModule
from src.pydanticai_module import *

import time
import textstat
import pandas as pd


system_prompt = "You are an agent that based on a given ontology-based narrative, shall provide a new narrative that: \
    (a) uses an easier language than the original, (b) is shorter than the original, and (c) keeps the semantic meaning \
    of the original. "

#system_prompt = "You are an agent that shall solve a {task}."

# user_prompt = "{task} : Based on a given ontology-based narrative, shall provide a new narrative that: (a) uses an easier \
#     language than the original, (b) is shorter than the original, and (c) keeps the semantic meaning of the original."



if __name__ == "__main__":
    start_time = time.time()

    utils_object = utilsModule()
    test_utils_object = testUtilsModule()

    model_to_run = 'qwen3:30b' # qwen3:14b  |  qwen3:8b  |  qwen3:1.7b  |  qwen3:32b  |  gpt-oss:20b  
    specificity = 3
    dataset_name = 'collaborative_ssd_case_inspection'

    narratives_csv_file = "generated_c_narratives_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + ".csv"
    enhanced_narratives_csv_file = "generated_c_narratives_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" + model_to_run + ".csv"
    evaluation_results_file = "evaluation_results_multiple_plans_comparison_with_specificity_" \
        + str(specificity) + "_dataset_" + dataset_name + "_using_language_models_" + model_to_run + ".csv"

    narratives_dict = utils_object.create_dict_from_csv_file(\
        utils_object.csv_file_path + "/explanatory_narratives_cra/acxon_based", \
        narratives_csv_file, ",")
    
    enhanced_narratives_dict = dict()
    for k in narratives_dict.keys():
        enhanced_narratives_dict[k] = list()
    enhanced_narratives_dict['Modifications'] = list()
    enhanced_narratives_dict['Language model'] = list()

    results_dict = {"Pair ID": list(), "Dataset" : list(), "Language model" : list(), "Specificity" : list(), \
                     "Words": list(), "Readability score": list(), "Cosine similarity": list()}
    
    # agent
    ollama_model = OpenAIModel(
    model_name=model_to_run, provider=OpenAIProvider(base_url='http://localhost:11434/v1')
    )
    
    explanation_generation_agent = Agent(ollama_model, output_type=OntologyBasedExplanationOutput, system_prompt=system_prompt)
    #explanation_generation_agent = Agent(ollama_model, output_type=str, system_prompt=system_prompt) # it returns the complete text (as using the chat) good for debugging 
    #explanation_generation_agent = Agent(ollama_model, output_type=Union[OntologyBasedExplanationOutput, str], system_prompt=system_prompt) # it returns the text (Union uses smart mode by default)

    #number_of_explanations = 5
    number_of_explanations = len(narratives_dict["Explanation"])
    for i in range (0, number_of_explanations):    
        user_prompt = narratives_dict["Explanation"][i]
        complete_user_prompt = user_prompt 

        result = explanation_generation_agent.run_sync(user_prompt=complete_user_prompt, message_history=None) # history is None by default, but just in case
        final_time = time.time()
        #print('Result: ', result.output)
        #print('New explanation: ', type(result.output))
        #print('New explanation: ', result.output.narrative)

        enhanced_narratives_dict['Pair ID'].append(narratives_dict['Pair ID'][i])
        enhanced_narratives_dict['Explanation'].append(result.output.narrative)
        enhanced_narratives_dict['Modifications'].append(result.output.modifications)
        enhanced_narratives_dict['Language model'].append(model_to_run)

        results_dict['Pair ID'].append(narratives_dict['Pair ID'][i])
        results_dict['Dataset'].append(dataset_name)
        results_dict['Language model'].append(model_to_run)
        results_dict['Specificity'].append(specificity)
        results_dict['Words'].append(len(result.output.narrative.split()))
        ##results_dict['Readability score'].append(textstat.dale_chall_readability_score(result.output.narrative))
        ##results_dict['Readability score'].append(textstat.text_standard(result.output.narrative, float_output=True))
        results_dict['Readability score'].append(textstat.flesch_reading_ease(result.output.narrative))
        results_dict['Cosine similarity'].append(test_utils_object.compute_text_cosine_similarity(\
            [narratives_dict["Explanation"][i], result.output.narrative])) # item() turns the np.float into python float

        elapsed_time = final_time - start_time
        print('Execution number: ', i)
        print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    
    df = pd.DataFrame.from_dict(enhanced_narratives_dict, orient='columns')
    df.to_csv(utils_object.csv_file_path + "/" + enhanced_narratives_csv_file, index=False) # data frame

    df = pd.DataFrame.from_dict(results_dict, orient='columns')
    df.to_csv(utils_object.csv_file_path + "/" + evaluation_results_file, index=False) # data frame

    print("\n ·· TEST ·····\n Narrative number of words (aprox.): ", results_dict["Words"]) 
    print(" Narrative readability (score): ", results_dict["Readability score"]) 
    print(" Narrative cosine similary: ", results_dict["Cosine similarity"]) 
    print("\n") 