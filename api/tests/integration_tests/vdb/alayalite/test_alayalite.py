from core.rag.datasource.vdb.alayalite.alayalite_vector import AlayaliteConfig, AlayaliteVector
from tests.integration_tests.vdb.test_vector_store import AbstractVectorTest, get_example_text, setup_mock_redis

class AlayaliteVectorTest(AbstractVectorTest):
    def __init__(self):
        super().__init__()
        self.vector = AlayaliteVector(
            collection_name="test_collection",
            config=AlayaliteConfig(
                url="/home/yujie/dify_test/api/storage/test/alayalite"  # Adjust the URL as needed for your test setup
            ),
        )

    def search_by_vector(self):
        hits_by_vector = self.vector.search_by_vector(query_vector=self.example_embedding)
        assert len(hits_by_vector) == 1

    def search_by_full_text(self):
        hits_by_full_text = self.vector.search_by_full_text(query=get_example_text())
        assert len(hits_by_full_text) == 0

def test_alayalite_vector(setup_mock_redis):
    AlayaliteVectorTest().run_all_tests()