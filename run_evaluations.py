#!/usr/bin/env3 python3
"""
LangSmith Evaluation Script for RAG System

This script runs evaluations on the AI Engineering Report RAG system using
custom correctness evaluator and LangSmith.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

from langsmith import Client, wrappers
import pandas as pd
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

#QA
inputs = [
    "For customer-facing applications, which company's models dominate the top rankings?",
    "What percentage of respondents are using RAG in some form?",
    "How often are most respondents updating their models?",
]

outputs = [
    "OpenAI models dominate, with 3 of the top 5 and half of the top 10 most popular models for customer-facing apps.",
    "70% of respondents are using RAG in some form.",
    "More than 50% update their models at least monthly, with 17% doing so weekly.",

]

# Run this only once

def create_client():
    client = Client()
    dataset_name = "Reportngg"

    # Store
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Input and expected output pairs for AgenticAIReport",
    )

    examples = [
        {
            "inputs": {"question": q},
            "outputs": {"answer": a},
        }
        for q, a in zip(inputs, outputs)
    ]
    client.create_examples(
        examples=examples,
        dataset_id=dataset.id,
    )
    return client

# Simple file adapter for local file paths
class LocalFileAdapter:
    """Adapter for local file paths to work with ChatIngestor."""
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.name = self.path.name

    def getbuffer(self) -> bytes:
        return self.path.read_bytes()

def answer_ai_report_question(
    inputs: dict,
    data_path: str = r"C:\LLMOPS\data\\2025_AI_Engineering_Report.txt",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 5,
    search_type: str = "mmr",
    fetch_k: int = 20,
    lambda_mult: float = 0.5) -> dict:
        
        """
        Answer questions about the AI Engineering Report using RAG.
        
        Args:
            inputs: Dictionary containing the question, e.g, {"question": "What is RAG?"}
            data_path: Path to the AI Engineering Report text file
            chunk_size: Size of the text chunks for splitting
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve

        Returns:
            Dictionary with the answer, e.g, {"answer": "RAG stands for..."}
        """

        try:
            # Extract question from inputs
            question = inputs.get("question","")
            if not question:
                return {"answer": "No question provided"}

            # Check if file exists
            if not Path(data_path).exists():
                return {"answer": f"Data File Not Found: {data_path}"}

            # Create file adapter
            file_adapter = LocalFileAdapter(data_path)

            # Build index using ChatIngestor
            ingestor = ChatIngestor(
                temp_base="data",
                faiss_base="faiss_index",
                use_session_dirs=True
            )

            # Building retriever
            ingestor.build_retriever(
                uploaded_files=[file_adapter],
                search_type=search_type,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                k=k
            )

            # Get session ID and index path
            session_id = ingestor.session_id
            index_path = f"faiss_index/{session_id}"

            # Create RAG instance and load retriever
            rag = ConversationalRAG(session_id=session_id)
            rag.load_retriever_from_faiss(
                index_path=index_path,
                k=k,
                index_name=os.getenv("FAISS_INDEX_NAME","index"),
                search_type=search_type,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )

            # Get answer
            answer = rag.invoke(question, chat_history=[])

            return {"answer": answer}

        except Exception as e:
            return {"answer": f"Error: {str(e)}"}

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:o3-mini",
        feedback_key="correctness",
    )
    return evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )

def run_evaluation(
    data: str = "Reportngg",
    evaluators: List = [correctness_evaluator],
    experiment_prefix: str = "test-Reportngg",
    max_concurrency: int = 2,
    target = None):

        experiment_results = create_client().evaluate(
        answer_ai_report_question,
        data=data,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
        )

        return experiment_results

if __name__ == '__main__':
    run_evaluation()


