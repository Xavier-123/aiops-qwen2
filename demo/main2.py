import asyncio

from dotenv import dotenv_values
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.core.tools import ToolMetadata, QueryEngineTool

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.openai_like import OpenAILike as OpenAI
from llama_index.legacy.llms import OpenAILike as OpenAI
from qdrant_client import models
from tqdm.asyncio import tqdm

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval, generation_with_knowledge_retrieval_sub

from custom.transformation import CustomQueryEngine, CustomSentenceTransformerRerank
# from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
# from llama_index.postprocessor import RankGPTRerank
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank


import pickle

from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)

import torch


async def main():
    torch.cuda.set_device(0)
    config = dotenv_values(".env")
    print(config)
    treedict = pickle.load(open('ftree.bin', 'rb'))
    print(treedict[list(treedict.keys())[0]])
    abbdict = pickle.load(open('abbdict.bin', 'rb'))

    # ==================================
    # Q = QueryEngineTool.from_defaults(CustomQueryEngine(llm='test',callback_manager=None))
    # print(Q.call('hi'))
    # ==================================

    abbreviation = pickle.load(open('abbreviate.bin', 'rb'))
    print(abbreviation[list(abbreviation.keys())[0]])

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager
    # 初始化 LLM 嵌入模型 和 Reranker

    # llm = Ollama(
    #    model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
    # )

    llm = OpenAI(
        api_key=config["GLM_KEY"],
        # model="glm-3-turbo",
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )

    rerank = []
    # Re-Rank the top 3 chunks based on the gpt-3.5-turbo-0125 model
    rerank.append(SentenceTransformerRerank(model=r"F:\inspur\EMBEDDING_MODEL\Xorbits\bge-reranker-large", top_n=5))
    # rerank.append(RankGPTRerank(
    #     top_n=5,
    #     llm=llm
    # ))

    # ==================================
    # llm_predictor = LLMPredictor(llm=llm)
    # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    # ==================================

    embeding = HuggingFaceEmbedding(
        model_name=r"F:\inspur\EMBEDDING_MODEL\m3e-base",    # 768
        # model_name=r"F:\inspur\EMBEDDING_MODEL\m3e-small", # 512
        # model_name="xrunda/m3e-base",
        cache_folder="./",
        embed_batch_size=256,
        # max_length =768 ,
        # encode_kwargs = {'normalize_embeddings': True},
        query_instruction='为这个句子生成表示以用于检索相关文章：'

    )
    print(embeding.max_length)
    Settings.embed_model = embeding

    print('Init pipeline and vector store')
    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, storepath='./vector', reindex=False, )

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )
    print(collection_info.points_count)

    if collection_info.points_count == 0:
        data = read_data(config["DATA_DIR"] or "data")
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store, themetree=treedict)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=30000),
        )
        print(f"data length: {len(data)}")
        # Update collection info !
        collection_info = await client.get_collection(
            config["COLLECTION_NAME"] or "aiops24"
        )
        print(f"points count:{collection_info.points_count}")

    retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=100)

    # ==================================
    # index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank] )
    # query_engine_tools = [
    # QueryEngineTool(
    #    query_engine=query_engine, 
    #    metadata=ToolMetadata(name='lyft_10k', description='Provides information about Lyft financials for year 2021')
    # ),
    # ]
    # s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

    # response = s_engine.query('RCP策略决策包括哪些策略类型?')
    # print('TEST>>>>>>>>>>>>>>>>>>',response)
    # ==================================

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        # result = await generation_with_knowledge_retrieval_sub(
        print("query：", query)
        result = await generation_with_knowledge_retrieval(
            query["query"], retriever, llm,
            abbreviation=abbreviation,
            debug=True,
            reranker=rerank,
            query_theme=query["document"],
            abbdict=abbdict,
        )
        results.append(result)
        break

    # 处理结果
    # save_answers(queries, results, "submit_result.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
