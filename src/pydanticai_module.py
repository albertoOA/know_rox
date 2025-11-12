#!/usr/bin/env python3
# coding=utf-8
# Author: Alberto Olivares Alarcos <aolivares@iri.upc.edu>, Institut de Robòtica i Informàtica Industrial, CSIC-UPC

from pydantic import BaseModel, Field
from typing import Union

from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class OntologyBasedExplanationOutput(BaseModel):
    narrative: str = Field(description='enhanced narrative generated from an ontology-based narrative')
    modifications: str = Field(description='summary of the changes in the output explanation with respect to the initial ontology-based narrative')