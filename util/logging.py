import os
import numpy as np

class Logger():
    def __init__(self, level='warning', console=True, file=None, console_formatter='{}', file_formatter='{}'):
        self.levels = {'debug':10, 'info':20, 'warning':30, 'error':40, 'critical':50}
        if level not in self.levels:
            raise ValueError('level {} doesnot exist'.format(level))
        self._level = self.levels[level]
        self._console = console
        self._file = file
        if self._file != None:
            if not os.path.exists(self._file):
                os.mknod(self._file)

        self._console_formatter = console_formatter
        self._file_formatter = file_formatter

    def log(self, message):
        if self._console:
            print(self._console_formatter.format(message))
        if self._file:
            with open(self._file,'a') as f:
                f.write(self._file_formatter.format(message))
                #f.write('\n')

    def debug(self, message):
        if self._level <= 10:
            self.log(message)

    def info(self, message):
        if self._level <= 20:
            self.log(message)

    def warning(self, message):
        if self._level <= 30:
            self.log(message)

    def error(self, message):
        if self._level <= 40:
            self.log(message)

    def critical(self, message):
        if self._level <= 50:
            self.log(message)

class LossLogger():
    def __init__(self) -> None:
        self.loss = None
        self.loss_count = None

    def get_last_average(self, n=0):
        average = {}
        for k,v in self.loss.items():
            average[k] = np.array(v[-n:]).sum() / np.array(self.loss_count[-n:]).sum()
        return average

    def clear(self):
        self.loss = None
        self.loss_count = None

    def update(self, loss_dict, count=1):
        if self.loss is None:
            self.loss = {}
            self.loss_count = []
            for k in loss_dict.keys():
                self.loss[k] = []

        for k,v in loss_dict.items():
            if k not in self.loss:
                self.loss[k] = []
            self.loss[k].append(v)
        self.loss_count.append(count)

        #if self.loss_average is None:
        #    self.loss_average = {}
        #    for k in loss_dict.keys():
        #        self.loss_average[k] = 0

        #self.loss_average_count += count
        #for k,v in loss_dict.items():
        #    self.loss_average[k] += v