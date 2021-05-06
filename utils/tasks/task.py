import datetime
from abc import ABC, abstractmethod


class Task(ABC):
    """
    Abstract class used to represent the tasks run by the simulator
    """
    def __init__(self, name, description):
        """

        :param name: the name of the task
        :type name: str
        :param description: the description of the task
        :type description: str
        """
        self.name = name
        self.description = description
        self.verbose = 0
        self.ready = False
        self.metadata = {
            'NAME': self.name,
            'DESCRIPTION': self.description,
            'EXECUTION_DATE': 'never'
        }

    @abstractmethod
    def run(self, force):
        pass

    @abstractmethod
    def config(self, *args):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def save(self, folder, override):
        pass

    @abstractmethod
    def plot(self, plot_number):
        pass

    def _print(self, text: str):
        if self.verbose != 0:
            print(text)

    def _finalize_execution(self):
        self.metadata['EXECUTION_DATE'] = str(datetime.datetime.now())
        self.ready = True
