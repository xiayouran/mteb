# -*- coding: utf-8 -*-
# Author  : liyanpeng
# Email   : yanpeng.li@cumt.edu.cn
# Datetime: 2025/4/24 18:51
# Filename: bysearch_models.py
from __future__ import annotations

from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader
from mteb.models.bge_models import bge_full_data

bysearch_zh_datasets = {
    "BQ": ["train"],
    "LCQMC": ["train"],
    "PAWSX": ["train"],
    "STS-B": ["train"],
    "DuRetrieval": ["train"],
    "AFQMC": ["train"],
    "Cmnli": ["train"],
    "Ocnli": ["train"],
    "T2Retrieval": ["train"],
    "T2Reranking": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
}

bysearch_embedding_ckpt_7500_no_prompt = ModelMeta(
    name="PeopleAI/BySearch-Embedding-checkpoint-7500_no_prompt",
    revision="",
    release_date="2025-04-24",
    languages=[
        "zho-Hans",
    ],
    loader=partial(
        sentence_transformers_loader,
        model_name="PeopleAI/BySearch-Embedding-checkpoint-7500_no_prompt",
    ),
    max_tokens=512,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license="apache-2.0",
    reference="https://huggingface.co/PeopleAI/BySearch-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets={
        **bge_full_data,
        **bysearch_zh_datasets,
    },
    public_training_code=None,
    public_training_data=None,
)
bysearch_embedding_ckpt_11000 = ModelMeta(
    name="PeopleAI/BySearch-Embedding-checkpoint-11000",
    revision="",
    release_date="2025-04-24",
    languages=[
        "zho-Hans",
    ],
    loader=partial(
        sentence_transformers_loader,
        model_name="PeopleAI/BySearch-Embedding-checkpoint-11000",
    ),
    max_tokens=512,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license="apache-2.0",
    reference="https://huggingface.co/PeopleAI/BySearch-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets={
        **bge_full_data,
        **bysearch_zh_datasets,
    },
    public_training_code=None,
    public_training_data=None,
)
bysearch_embedding_ckpt_11000_prompt = ModelMeta(
    name="PeopleAI/BySearch-Embedding-checkpoint-11000-prompt",
    revision="",
    release_date="2025-04-24",
    languages=[
        "zho-Hans",
    ],
    loader=partial(
        sentence_transformers_loader,
        model_name="PeopleAI/BySearch-Embedding-checkpoint-11000-prompt",
    ),
    max_tokens=512,
    embed_dim=1536,
    open_weights=False,
    n_parameters=None,
    memory_usage_mb=None,
    license="apache-2.0",
    reference="https://huggingface.co/PeopleAI/BySearch-Embedding",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=False,
    training_datasets={
        **bge_full_data,
        **bysearch_zh_datasets,
    },
    public_training_code=None,
    public_training_data=None,
)

