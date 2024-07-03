from typing import Sequence
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode
from collections import defaultdict
import re


class CustomFilePathExtractor(BaseExtractor):
    last_path_length: int = 4
    themetree: dict = {}

    def __init__(self, last_path_length: int = 4, **kwargs, ):
        super().__init__(last_path_length=last_path_length, **kwargs)

    #        print(self.themetree)

    @classmethod
    def class_name(cls) -> str:
        return "CustomFilePathExtractor"

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        metadata_list = []
        for node in nodes:
            # node.metadata["file_path"] = "/".join(
            #    node.metadata["file_path"].split("/")[-self.last_path_length :]
            # )
            node.metadata["theme"] = node.metadata["file_path"].split("/")[11]
            node.metadata["sub_theme"] = node.metadata["file_path"].split("/")[12]
            node.metadata["fileid"] = node.metadata["file_path"].split("/")[-1].split('.txt')[0]
            node.metadata["themetree"] = ""
            fkey = f'{node.metadata["theme"]}/{node.metadata["sub_theme"]}/{node.metadata["fileid"]}'
            if fkey in self.themetree.keys():
                node.metadata["themetree"] = self.themetree[fkey]

            repp = re.compile('[0-9a-zA-Z]+')
            node.metadata["abbreviation"] = list(set(repp.findall(node.text)))
            metadata_list.append(node.metadata)
        return metadata_list


class CustomTitleExtractor(BaseExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomTitleExtractor"

    # 将Document的第一行作为标题
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        try:
            document_title = nodes[0].text.split("\n")[0]
            last_file_path = nodes[0].metadata["file_path"]
        except:
            document_title = ""
            last_file_path = ""
        metadata_list = []
        for node in nodes:
            if node.metadata["file_path"] != last_file_path:
                document_title = node.text.split("\n")[0]
                last_file_path = node.metadata["file_path"]
            node.metadata["document_title"] = document_title
            metadata_list.append(node.metadata)

        return metadata_list


from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import QueryBundle
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import CompletionResponse

from custom.template import RW_TEMPLATE, OW_TEMPLATE
import re


def queryGenerations(query_str, llm, num_queries=1):
    fmt_rw_prompt = PromptTemplate(RW_TEMPLATE)
    fmt_ow_prompt = PromptTemplate(OW_TEMPLATE)
    rw_response = llm.predict(fmt_rw_prompt, num_queries=num_queries, query_str=query_str)
    ow_response = llm.predict(fmt_ow_prompt, num_queries=num_queries, query_str=query_str)
    # assume LLM proper put each query on a newline
    rw_queries = [re.sub('^\d\.', '', i) for i in rw_response.split("\n")[-2:]]
    ow_queries = [re.sub('^\d\.', '', i) for i in ow_response.split("\n")[-2:]]
    # print(f"query_str:\n{query_str}")
    # print(f"Generated rw_queries:\n{rw_queries}")
    # print(f"Generated ow_queries:\n{ow_queries}")

    return rw_queries, ow_queries


def queryReplace(query_str, abbdict, abblist):
    found_key = {}
    for i in abblist:
        if i in abbdict.keys():
            found_key[i] = [k for k in abbdict[i]]
    gen_ques = []
    new_query = query_str
    for i in found_key.keys():
        for j in found_key[i]:
            if len(j) > 1:
                new_query = new_query.replace(i, j[1])
    return [new_query]


from llama_index.core.postprocessor import LongContextReorder


class CustomQueryEngine(BaseQueryEngine):
    def __init__(self, llm, retriever, filters, qa_template, reranker, debug, abbdict, **kwargs):
        self.llm = llm
        self.retriever = retriever
        self.filters = filters
        self.reranker = reranker
        self.debug = debug
        self.qa_template = qa_template
        self.abbdict = abbdict
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "CustomQueryEngine"

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        print("query_bundle:", query_bundle)
        print("self.filters:", self.filters)
        # 先用原始query做一次，向量查询
        node_with_scores = await self.retriever.aretrieve_cc(query_bundle, qdrant_filters=self.filters)
        repp = re.compile('[0-9a-zA-Z]+')
        abblist = repp.findall(query_bundle.query_str)
        print("query_str:", query_bundle.query_str)

        gen_ques = []
        if True:
            gen_ques, gen_ques2 = queryGenerations(query_bundle.query_str, self.llm)
        print("queryGenerations gen_ques:", gen_ques)
        print("queryGenerations gen_ques2:", gen_ques2)
        # if True:
        #     gen_ques = queryReplace(query_bundle.query_str, self.abbdict, abblist)
        # print("queryReplace gen_ques:", gen_ques)

        # 再用生成的新query，向量查询
        if True:
            known_id = [node.id_ for inode, node in enumerate(node_with_scores)]

            # 将问题差分为小问题，一次查询
            for iques in gen_ques:
                tmp_node_with_score = await self.retriever.aretrieve_cc(QueryBundle(query_str=iques), qdrant_filters=self.filters)
                # Drop Duplicates
                for inode, node in enumerate(tmp_node_with_score):
                    if node.id_ not in known_id:
                        node_with_scores.append(node)
                        known_id.append(node.id_)

            # 将问题重写，再查询
            for iques in gen_ques:
                tmp_node_with_score = await self.retriever.aretrieve_cc(QueryBundle(query_str=iques), qdrant_filters=self.filters)
                # Drop Duplicates
                for inode, node in enumerate(tmp_node_with_score):
                    if node.id_ not in known_id:
                        node_with_scores.append(node)
                        known_id.append(node.id_)


        print("-"*100)
        print("node_with_scores:")
        print(len(node_with_scores), node_with_scores[0])
        # print("known_id:", known_id)
        print("-" * 100)

        if self.reranker:
            try:
                node_with_scores = self.reranker[0].postprocess_nodes(node_with_scores, query_bundle)
            except Exception as e:
                print("e:", e)
                node_with_scores = self.reranker[1].postprocess_nodes(node_with_scores, query_bundle)
                # if self.debug:
                #    print(f"reranked:\n{node_with_scores}\n------")
                node_with_scores = self.reranker[1].postprocess_nodes(node_with_scores, query_bundle)

        if False:
            for i in node_with_scores:
                tmp_node_with_score = await self.retriever.aretrieve_cc(
                    QueryBundle(query_str=query_budle.query_str + i.text), qdrant_filters=self.filters)

        # 长上下文阅读器
        lreorder = False
        if lreorder:
            LCreorder = LongContextReorder()
            node_with_scores = LCreorder.postprocess_nodes(node_with_scores)

        if self.debug:
            # node.metadata['theme'] 文件夹名称
            print('Basic info:', '\n'.join(
                [f"背景知识{inode}:\n{node.score}{node.metadata['document_title']}, {node.metadata['theme']},"
                 f"{node.metadata['sub_theme']},{node.metadata['fileid']},{node.metadata['themetree']}"
                 for inode, node in enumerate(node_with_scores)]))

        context_str = ''

        if False:
            filelist = defaultdict(list)
            for inode, node in enumerate(node_with_scores):
                filelist[node.metadata['file_path']].append(inode)
            print(filelist)
            if node_with_scores[0].score > 0.9:
                to_read = [node_with_scores[0].metadata['file_path']]
                todelnode = [0]
            else:
                to_read = []
                todelnode = []

            for i in filelist.keys():
                if len(filelist[i]) > 1:
                    if i not in to_read:
                        to_read.append(i)
                    # Hitted, cancel others!
                    if i == to_read[0]:
                        to_read = [i]
                        node_with_scores = []
                        break
                    todelnode.extend(filelist[i])
                    print(f'{i} is hitted for several times')

            print(todelnode)

            node_with_scores = [temp_data for index, temp_data in enumerate(node_with_scores) if index not in todelnode]
            print(f'After Processing:{node_with_scores}')

            for iftoread in to_read:
                with open(iftoread, 'r') as fp:
                    context_str = ''.join(fp.readlines())

        context_str += "\n".join(
            [f"背景知识{inode}:\n{node.text}" for inode, node in enumerate(node_with_scores)]
        )
        context_str = context_str.replace('~', '')
        context_str = context_str.replace('$', '')
        context_str = context_str.replace('>>>', '')
        context_str = context_str.replace('>>:', '')

        fmt_qa_prompt = PromptTemplate(self.qa_template).format(
            context_str=context_str, query_str=query_bundle.query_str
        )
        if self.debug:
            print('QUERY>>>', query_bundle.query_str, '\n', context_str)  # 问题，上下文
        try:
            ret = await self.llm.acomplete(fmt_qa_prompt)
        except Exception as e:
            print(f'Request failed:{context_str}')
            ret = CompletionResponse(text='不确定')
        return Response(ret.text)

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        return

    def _get_prompt_modules():
        return {}


from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.question_gen.types import SubQuestion, SubQuestionList
from llama_index.core.question_gen.prompts import build_tools_text
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.output_parsers.base import StructuredOutput
from typing import List, cast


class CustomQuestionGenerator(LLMQuestionGenerator):
    def generate(
            self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = self._llm.predict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        print(parse.parsed_output)
        return parse.parsed_output

    async def agenerate(
            self, tools: Sequence[ToolMetadata], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            tools_str=tools_str,
            query_str=query_str,
        )

        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        parse = cast(StructuredOutput, parse)
        Questions = [SubQuestion(sub_question=query.query_str, tool_name='query_engine_tool')] + parse.parsed_output
        print(Questions)
        return parse.parsed_output


from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.utils import infer_torch_device


class CustomSentenceTransformerRerank(SentenceTransformerRerank):

    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]

        with self.callback_manager.event(
                CBEventType.RERANKING,
                payload={
                    EventPayload.NODES: nodes,
                    EventPayload.MODEL_NAME: self.model,
                    EventPayload.QUERY_STR: query_bundle.query_str,
                    EventPayload.TOP_K: self.top_n,
                },
        ) as event:
            scores = self._model.predict(query_and_nodes)

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes_tmp = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                            : self.top_n
                            ]
            threshold = 0.5
            new_nodes = []
            for i in new_nodes_tmp:
                if i.score > threshold:
                    new_nodes.append(i)

            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
