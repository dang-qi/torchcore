import os


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
