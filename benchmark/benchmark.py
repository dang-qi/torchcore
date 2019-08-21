import datetime
from tools import Logger
from .bench_criterias import bench_criterias

class Benchmark :
    def __init__( self, cfg, criterias ):
        self._criterias = criterias
        self._cfg = cfg
        self.init_logger()
        self.init_bench()

    def init_bench(self):
        self._bench = []
        for criteria in self._criterias:
            if criteria not in bench_criterias:
                raise ValueError('The criteria {} is not supported, the supported ones are: {}'.format(criteria, list(bench_criterias.keys())))
            else:
                self._bench.append(bench_criterias[criteria](self._cfg, self._logger))

    def init_logger(self):
        bench_path = self._cfg.path['BENCHMARK_LOG']

        console_formatter = '{} {{}}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self._logger = Logger(level='info', file=bench_path, console=False, console_formatter=console_formatter)

        #self._logger.info('benchmark log path is: {}'.format(bench_path))
        print('The benchmark log path is {}'.format(bench_path))

    def update( self, targets, pred ):
        for bench in self._bench:
            bench.update(targets, pred)

    def summary( self ):
        for bench in self._bench:
            bench.summary()

    def update_parameters(self, parameters):
        for bench in self._bench:
            bench.update_parameters(parameters)