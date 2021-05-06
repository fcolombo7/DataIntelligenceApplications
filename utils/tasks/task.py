import datetime
import json
import os
import shutil
from abc import ABC, abstractmethod
from zipfile import ZipFile


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
        self.result = {}

    @abstractmethod
    def run(self, force):
        pass

    @abstractmethod
    def config(self, *args):
        pass

    def load(self, filename):
        with ZipFile(filename, 'r') as archive:
            with archive.open('metadata.json') as f:
                self.metadata = json.load(f)
                print(self.metadata)
            with archive.open('content.json') as f:
                self.result = json.load(f)
                print(self.result)
        self.ready = True

    def save(self, folder='simulations_results', override=False):
        assert os.path.isdir(folder) and self.ready
        cur_dir = os.getcwd()
        os.chdir(folder)
        temp_dir_path = f'temp_{self.name}'
        os.mkdir(temp_dir_path)
        metadata = json.dumps(self.metadata)
        print(metadata)
        content = json.dumps(self.result)
        print(content)
        with open(os.path.join(temp_dir_path, 'metadata.json'), 'w') as f:
            f.write(metadata)
        with open(os.path.join(temp_dir_path, 'content.json'), 'w') as f:
            f.write(content)
        # Create a ZipFile Object
        filename = f'result_{self.name}'
        if not override:
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
            os.chdir(folder)
        # delete the temp dir
        shutil.rmtree(temp_dir_path)
        os.chdir(cur_dir)

    @abstractmethod
    def plot(self, plot_number):
        pass

    def _print(self, text: str):
        if self.verbose != 0:
            print(text)

    def _finalize_execution(self):
        self.metadata['EXECUTION_DATE'] = str(datetime.datetime.now())
        self.ready = True
