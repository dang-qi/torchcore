from .bench_base import BenchBase
from .category_accuracy import CategoryAccuracy
from .pair_match_accuracy import PairMatchAccuracy

bench_criterias = {}
bench_criterias['category_accuracy'] = CategoryAccuracy
bench_criterias['pair_match_accuracy'] = PairMatchAccuracy