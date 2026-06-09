# KNOW-ROX
Knowledge-based framework for robot ontology-based explanations. This framework proposes the integration of sound ontology-based knowledge and the power of language models to generate truthful and natural (human-friendly) explanations. 


### Python3 virtual environment configuration and dependencies

In the repository, we already provide a virtual environment but it was built for our computer and it will probably not work in yours. You can easily delete it and create and configure your own environment executing the following commands in a terminal (note that python3-venv needs to bee installed).

```
cd <know_rox_folder>/python_environment
python3 -m venv know_rox_venv
source know_rox_venv/bin/activate

python3 -m pip install pandas ollama pydantic-ai textstat scikit-learn transformers[torch] transformers sentence-transformers deepeval
```

It is also necessary to install the python modules included in the source folder. 

```
cd <know_rox_folder>

python3 -m pip install .
```

It is also possible to install them in 'editable' (development) mode if you want to make some modifications in those modules. 

```
cd <know_rox_folder>

python3 -m pip install -e .
```

### Setup for evaluation
Part of the evaluation of the explanations consists in computing their semantic similarity with respect to the original ontology-based narratives. This is done using different estrategies/metrics: direct cosine similarity on the text, cross-encoder similarity, and LLM-as-a-judge. For the last one, we are using the deepEval library, which allows using models from Ollama, but comes with a pre-selected list of available models. Hence, if you want to use a specific model for the evaluation (e.g. 'gpt-oss:20b') it is necessary to set it up in the terminal (CLI).

```
deepeval set-ollama --model=gpt-oss:20b
```



### Downloading files from an external github repository (explanatory_narratives_cra)
In order to avoid creating a duplicate of the logical rules to compare planes formalized and implemented in [explanatory_narratives_cra](https://github.com/albertoOA/explanatory_narratives_cra), ***know-rox*** includes a script to download and update the pertinent files. It is possible to program a github action to download them regularly, but for now it will be a manual process. Note that it is necessary to have subversion (svn) installed before running the script. 

```
cd <know_rox_folder>
chmod +x scripts/sh/update_shared_files_ontology_based_narratives.sh 
./scripts/sh/update_shared_files_ontology_based_narratives.sh
``` 


### Generating explanations using existing ontology-based narratives
In this example, we are running a script that generates enhanced ontology-based narratives leveraging the capabilities of language models. The constructed explanations are expected to be shorter, easier to understand while keeping the original semantic meaning. Hence, an evaluation is performed using three metrics: number of words, readability score, and semantic (cosine) similarity. 

```
cd <know_rox_folder>
python3 scripts/explanation_generation_from_ontology_based_narrative.py
``` 

The example script may be modified to select the language model to use, and the original ontology-based narratives (dataset and specificity). Note that the language model is expected to be listed within the available models in ***ollama***, and it requires *tool* support. 