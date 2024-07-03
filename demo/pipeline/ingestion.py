from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.schema import Document, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Self defined sentence splitter
from llama_index.core.node_parser import SentenceSplitter
#from pipeline.sentencecus import SentenceSplitter

from llama_index.core.node_parser import MarkdownNodeParser

from custom.template import SUMMARY_EXTRACT_TEMPLATE
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor


def read_data(path: str = "data") -> list[Document]:
    reader = SimpleDirectoryReader(
        input_dir=path,
        recursive=True,
        exclude=[
             '.*/nodes/.*'
        ],
        required_exts=[
            ".txt",
            ".md",
        ],
    )
    return reader.load_data()


def build_pipeline(
    llm: LLM,
    embed_model: BaseEmbedding,
    template: str = None,
    vector_store: BasePydanticVectorStore = None,
    themetree:dict = None
) -> IngestionPipeline:
    transformation = [
        #SentenceSplitter(chunk_size=1024, chunk_overlap=100, paragraph_separator='~',secondary_paragraph_chunking_regex='>>>',secondary_chunking_regex='>>:'),
        MarkdownNodeParser(),
        #SentenceSplitter(chunk_size=256, chunk_overlap=100),# paragraph_separator='###'),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4,themetree=themetree, metadata_mode=MetadataMode.EMBED),
        # SummaryExtractor(
        #     llm=llm,
        #     metadata_mode=MetadataMode.EMBED,
        #     prompt_template=template or SUMMARY_EXTRACT_TEMPLATE,
        # ),
        embed_model,
    ]

    return IngestionPipeline(transformations=transformation, vector_store=vector_store)

from llama_index.vector_stores.qdrant.utils import default_sparse_encoder
async def build_vector_store(
    config: dict, reindex: bool = False,
    storepath='./vector'
) -> tuple[AsyncQdrantClient, QdrantVectorStore]:
    client = AsyncQdrantClient(
        # url=config["QDRANT_URL"],
        # location=":memory:"
        path=storepath
    )
    if reindex:
        print(f"REINDEX THE COLLECTION: {config['COLLECTION_NAME']}")
        try:
            await client.delete_collection(config["COLLECTION_NAME"] or "aiops24")
        except UnexpectedResponse as e:
            print(f"Collection not found: {e}")

        try:
            await client.create_collection(
                collection_name=config["COLLECTION_NAME"] or "aiops24",
                #vectors_config={
                #    'text-dense':models.VectorParams(
                #        size=config["VECTOR_SIZE"] or 1024, distance=models.Distance.DOT
                #    )
                #},
        
                #sparse_vectors_config={
		#    "text-sparse-new": models.SparseVectorParams(
                #        index=models.SparseIndexParams(
                #            on_disk=False,
                #        )
                #    )
                #},

                vectors_config=models.VectorParams(
                        size=config["VECTOR_SIZE"] or 1024, 
                        #distance=models.Distance.DOT
                        distance=models.Distance.DOT
                    )
            )
        except UnexpectedResponse:
            print("Collection already exists")
    return client, QdrantVectorStore(
        aclient=client,
        collection_name=config["COLLECTION_NAME"] or "aiops24",
        parallel=4,
        batch_size=32,
        #enable_hybrid=True,
        #sparse_doc_fn = default_sparse_encoder("naver/efficient-splade-VI-BT-large-doc"),
        #sparse_query_fn =default_sparse_encoder("naver/efficient-splade-VI-BT-large-query"),
    )
