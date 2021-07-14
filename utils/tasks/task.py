import datetime
import json
import os
import shutil
from abc import ABC, abstractmethod
from zipfile import ZipFile

from data_generators.standard_generator import StandardDataGenerator


class Task(ABC):
    """
    Abstract class used to represent the tasks run by the simulator
    """
    def __init__(self, name: str, description: str, src: str, verbose: int):
        """
        Class constructor
        :param name: the name of the task
        :param description: the description of the task
        :param verbose: level of verbose. (0 or 1)
        """
        self.name = name
        self.description = description
        self.verbose = verbose
        self.data_src = src
        self.data_generator = StandardDataGenerator(self.data_src)
        self.ready = False
        self.metadata = {
            'NAME': self.name,
            'DESCRIPTION': self.description,
            'EXECUTION_DATE': 'never',
            'SRC': self.data_src
        }
        self.T = None
        self.n_experiments = None
        self.learners_to_test = None
        self.result = {}

    @abstractmethod
    def run(self) -> None:
        """
        Execution of the simulation.
        """
        pass

    @abstractmethod
    def config(self, *args) -> None:
        """
        Method used to configure the hyper paramenter of the simulation
        :param args: hyperparameter of the simulation
        :return:
        """
        pass

    @abstractmethod
    def plot(self, plot_number: int, figsize: (float, float)) -> None:
        """
        Plot the result of the simulation.
        :param plot_number: Which plot to show.
        :param figsize: dimension of the figure.
        """
        pass

    def load(self, filename: str) -> None:
        """
        Load the result computed by a previous simulation.
        :param filename: path to the zip file containing the simulation to load.
        """
        with ZipFile(filename, 'r') as archive:
            with archive.open('metadata.json') as f:
                self.metadata = json.load(f)
                # print(self.metadata)
            with archive.open('content.json') as f:
                self.result = json.load(f)
                # print(self.result)
        self.ready = True
        self._print(f"Simulation's result loaded from `{filename}`.")

    def save(self, folder='simulations_results', overwrite=False) -> str:
        """
        Save the result computed by the current simulation.
        :param folder: output folder.
        :param overwrite: if True overwrite the results already available (if present), otherwise it saves another file.
        """
        assert os.path.isdir(folder) and self.ready
        cur_dir = os.getcwd()
        os.chdir(folder)
        temp_dir_path = f'temp_{self.name}'
        os.mkdir(temp_dir_path)
        metadata = json.dumps(self.metadata)
        content = json.dumps(self.result)
        with open(os.path.join(temp_dir_path, 'metadata.json'), 'w') as f:
            f.write(metadata)
        with open(os.path.join(temp_dir_path, 'content.json'), 'w') as f:
            f.write(content)
        # Create a ZipFile Object
        filename = f'result_{self.name}'
        if not overwrite:
            count = 0
            while os.path.isfile(f'{filename}.zip'):
                count += 1
                if count == 1:
                    filename += '_'
                else:
                    filename = filename[:-1]
                filename += str(count)

        with ZipFile(f'{filename}.zip', 'w') as zipObj:
            # Add multiple files to the zip
            os.chdir(temp_dir_path)
            zipObj.write('metadata.json')
            zipObj.write('content.json')
            os.chdir('../')
        # delete the temp dir
        shutil.rmtree(temp_dir_path)
        os.chdir(cur_dir)
        self._print(f"The simulation's result is stored in `{folder}/{filename}.zip`.")
        path = f'{folder}/{filename}.zip'
        return path

    def _print(self, text: str) -> None:
        """
        Custom print according to the class verbose.
        :param text: String to be printed.
        """
        if self.verbose != 0:
            print(text)

    def _finalize_execution(self) -> None:
        """
        Append information to the metadata structure and make the result available to be plotted.
        """
        self.metadata['EXECUTION_DATE'] = str(datetime.datetime.now())
        self.ready = True
