from pydantic import BaseModel
from alayalite import Client
import json

class AlayaliteConfig(BaseModel):
    url: str

    def to_client_params(self):
        return {
            "url": self.url,
        }


from configs import dify_config
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from models.dataset import Dataset

class AlayaliteVector(BaseVector):
    def __init__(self, collection_name: str, config: AlayaliteConfig):
        super().__init__(collection_name)
        self.config = config
        self._collection_name = collection_name
        self.client = Client(**self.config.to_client_params())

    def get_type(self) -> str:
        return VectorType.ALAYALITE

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        if texts:
            self.create_collection(self._collection_name)
            self.add_texts(texts, embeddings, **kwargs)

    def create_collection(self, collection_name: str):
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        ids = self._get_uuids(documents)
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        collection = self.client.get_or_create_collection(self._collection_name)

        items = [
            (id, text, vector, metadata)
            for id, text, vector, metadata in zip(ids, texts, embeddings, metadatas)
        ]
        self.collection.upsert(items)

    def text_exists(self, id: str) -> bool:   
        result = self.collection.get_by_id([id])
        return bool(result and len(result) > 0)

    def delete_by_ids(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete_by_id(ids)

    def delete_by_metadata_field(self, key: str, value: str) -> None:

        all_items = self.collection.get_all()
        to_delete_ids = [item["id"] for item in all_items if item["metadata"].get(key) == value]
        if to_delete_ids:
            self.collection.delete_by_id(to_delete_ids)

    def search_by_vector(self, query_vector: list[float], **kwargs) -> list[Document]:
        limit = kwargs.get("top_k", 4)
        ef_search = kwargs.get("ef_search", 100)
        num_threads = kwargs.get("num_threads", 1)
        results = self.collection.batch_query([query_vector], limit, ef_search, num_threads)
        docs = []
        for result in results:
            doc = Document(
                page_content=result["text"],
                metadata=result.get("metadata", {}),
            )
            docs.append(doc)
        return docs

    def search_by_full_text(self, query: str, **kwargs) -> list[Document]:
        return []

    def delete(self) -> None:
        self.collection.delete()


class AlayaliteVectorFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> AlayaliteVector:
        if dataset.index_struct_dict:
            class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix.lower()

        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id).lower()
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.ALAYALITE, collection_name))

        return AlayaliteVector(
            collection_name=collection_name,
            config=AlayaliteConfig(url=dify_config.URL)
        )