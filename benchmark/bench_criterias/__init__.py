from .bench_base import BenchBase
from .category_accuracy import CategoryAccuracy
from .pair_match_accuracy import PairMatchAccuracy
from .retrieval_classification_accuracy import RetrievalClassificationAccuracy

bench_criterias = {}
bench_criterias['category_accuracy'] = CategoryAccuracy
bench_criterias['pair_match_accuracy'] = PairMatchAccuracy
bench_criterias['retrieval_classification_accuracy'] = RetrievalClassificationAccuracy