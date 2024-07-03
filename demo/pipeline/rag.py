from typing import List
import qdrant_client

from qdrant_client import QdrantClient, models

import re

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

from llama_index.core.llms.llm import LLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import (
    QueryBundle,
    PromptTemplate,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryType,
    TextNode,
)

from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

from custom.template import QA_TEMPLATE, QA_TEMPLATE_WHICH, QA_TEMPLATE_YN, QA_TEMPLATE2
from custom.transformation import CustomQueryEngine, CustomQuestionGenerator
from llama_index.core.base.response.schema import Response


class QdrantRetriever(BaseRetriever):
    def __init__(
            self,
            vector_store: QdrantVectorStore,
            embed_model: BaseEmbedding,
            similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        super().__init__()

    @dispatcher.span
    async def aretrieve_cc(self, str_or_query_bundle: QueryType, qdrant_filters=None) -> List[NodeWithScore]:
        self._check_callback_manager()
        dispatch_event = dispatcher.get_dispatch_event()

        dispatch_event(
            RetrievalStartEvent(
                str_or_query_bundle=str_or_query_bundle,
            )
        )
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        with self.callback_manager.as_trace("query"):
            with self.callback_manager.event(
                    CBEventType.RETRIEVE,
                    payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = await self._aretrieve(query_bundle=query_bundle, qdrant_filters=qdrant_filters)
                nodes = await self._ahandle_recursive_retrieval(
                    query_bundle=query_bundle, nodes=nodes
                )
                retrieve_event.on_end(
                    payload={EventPayload.NODES: nodes},
                )
        dispatch_event(
            RetrievalEndEvent(
                str_or_query_bundle=str_or_query_bundle,
                nodes=nodes,
            )
        )
        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle, qdrant_filters=None) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k,
            # mode = 'hybrid',
        )
        query_result = await self._vector_store.aquery(vector_store_query, qdrant_filters=qdrant_filters)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding, similarity_top_k=self._similarity_top_k
        )
        query_result = self._vector_store.query(vector_store_query)

        node_with_scores = []
        for node, similarity in zip(query_result.nodes, query_result.similarities):
            node_with_scores.append(NodeWithScore(node=node, score=similarity))
        return node_with_scores


async def generation_with_knowledge_retrieval(
        query_str: str,
        retriever: BaseRetriever,
        llm: LLM,
        # qa_template: str = QA_TEMPLATE,
        qa_template: str = QA_TEMPLATE2,
        reranker: list | None = None,
        debug: bool = False,
        query_theme: str = '',
        progress=None,
        abbreviation: dict = {},
        abbdict: dict = {},
) -> CompletionResponse:
    repp = re.compile('[0-9a-zA-Z]+')
    setlist = []
    themelist = []
    query_keys = repp.findall(query_str)
    print("query_keys:", query_keys)
    for i in query_keys:
        if i in abbreviation.keys():
            themeset = set()
            print(abbreviation[i])
            for j in abbreviation[i]:
                themeset.add(j[1])
            setlist.append(themeset)
    if len(setlist) > 0:
        theme = set.intersection(*setlist)
        if len(theme) == 0:
            theme = set.union(*setlist)
        themelist = list(theme)
        print(f'theme:{themelist}')

        # Form the filter based on questions
    filters = None
    conditions = []
    # if len(query_keys)>0:
    #    conditions = []
    #    for i in query_keys:
    #        conditions.append( models.FieldCondition(key='abbreviation',match=models.MatchValue(value=i)) )
    if query_theme != '':
        conditions.append(models.FieldCondition(key='theme', match=models.MatchValue(value=query_theme)))
        conditions.append(models.FieldCondition(key='theme', match=models.MatchValue(value=query_theme + '_sumary')))
    if len(conditions) > 0:
        filters = models.Filter(should=conditions)
        # filters = models.Filter(must=conditions)

    print("conditions:", conditions)
    print("filters:", filters)

    query_bundle = QueryBundle(query_str=query_str)

    # if "哪" in query_str:
    #    qa_template=QA_TEMPLATE_WHICH
    # if "是否" in query_str:
    #    qa_template=QA_TEMPLATE_YN

    query_engine = CustomQueryEngine(llm=llm, retriever=retriever, filters=filters, qa_template=qa_template,
                                     reranker=reranker, debug=debug, callback_manager=None, abbdict=abbdict)

    if False:
        print('Hyde')
        hyde = HyDEQueryTransform(include_original=True, llm=llm, hyde_prompt=PromptTemplate(
            '你是一个运维领域专家，请尝试回答以下问题：\n\n{context_str}\n'))

        hyde_query_engine = TransformQueryEngine(query_engine, hyde)

        query_result = await hyde_query_engine.aquery(query_bundle)
        ret = CompletionResponse(text=query_result.response)
    else:

        Q = QueryEngineTool.from_defaults(query_engine)
        query_result = await Q.acall(query_str)
        ret = CompletionResponse(text=query_result.content)

    if progress:
        progress.update(1)
    if debug:
        # print(f'Response: {ret}')
        print('Response: ------------------------------------------------------\n')
        print(ret)
    return ret


from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)


async def generation_with_knowledge_retrieval_sub(
        query_str: str,
        retriever: BaseRetriever,
        llm: LLM,
        qa_template: str = QA_TEMPLATE,
        reranker: list | None = None,
        debug: bool = False,
        progress=None,
        abbreviation: dict = {},
        abbdict: dict = {},
) -> CompletionResponse:
    repp = re.compile('[0-9a-zA-Z]+')
    setlist = []
    themelist = []
    query_keys = repp.findall(query_str)
    for i in query_keys:
        if i in abbreviation.keys():
            themeset = set()
            print(abbreviation[i])
            for j in abbreviation[i]:
                themeset.add(j[1])
            setlist.append(themeset)
    if len(setlist) > 0:
        theme = set.intersection(*setlist)
        if len(theme) == 0:
            theme = set.union(*setlist)
        themelist = list(theme)
        print(f'theme:{themelist}')

        # Form the filter based on questions
    filters = None
    # if len(query_keys)>0:
    #    conditions = []
    #    for i in query_keys:
    #        conditions.append( models.FieldCondition(key='abbreviation',match=models.MatchValue(value=i)) )
    #    filters = models.Filter(should=conditions)
    #    #filters = models.Filter(must=conditions)

    # print(filters)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager

    query_bundle = QueryBundle(query_str=query_str)

    Q = QueryEngineTool.from_defaults(
        CustomQueryEngine(llm=llm, retriever=retriever, filters=filters, qa_template=qa_template, reranker=reranker,
                          debug=debug, callback_manager=None, abbdict=abbdict))
    query_engine_tools = [Q]

    question_gen = CustomQuestionGenerator.from_defaults(llm=llm)
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools, llm=llm,
        question_gen=question_gen,
    )

    response = await query_engine.aquery(query_str)
    print(f"sub question query result: {response}")

    ret = CompletionResponse(text=response.response)

    if progress:
        progress.update(1)
    if debug:
        print(f'Response: {ret}')
    return ret
