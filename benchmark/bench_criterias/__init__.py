from .bench_base import BenchBase
from .category_accuracy import CategoryAccuracy
from .pair_match_accuracy import PairMatchAccuracy
from .retrieval_classification_accuracy import RetrievalClassificationAccuracy
from .top_k_retrieval_accuracy import TopKRetrievalAccuracy

bench_criterias = {}
bench_criterias['category_accuracy'] = CategoryAccuracy
bench_criterias['pair_match_accuracy'] = PairMatchAccuracy
bench_criterias['retrieval_classification_accuracy'] = RetrievalClassificationAccuracy
bench_criterias['top_k_retrieval_accuracy'] = TopKRetrievalAccuracy