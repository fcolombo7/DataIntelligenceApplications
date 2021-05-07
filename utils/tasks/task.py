import datetime
import numpy as np
import json
import os
import shutil
from abc import ABC, abstractmethod
from multiprocessing import Process, Lock, cpu_count, Manager
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
        self.T = None
        self.n_experiments = None
        self.learners_to_test = None
        self.result = {}

    @abstractmethod
    def _serial_run(self, process_number: int, n_experiments: int, collector, lock: Lock):
        pass

    @abstractmethod
    def config(self, *args):
        pass

    @abstractmethod
    def _finalize_run(self, collected_values):
        pass

    @abstractmethod
    def plot(self, plot_number):
        pass

    def run(self, force=False, parallelize=True, cores_number=-1):
        # check if the config method has been called
        assert self.T is not None and self.learners_to_test is not None and self.n_experiments is not None

        if not force and self.ready:
            self._print('Warning: The task was already executed, so the result is available.')
            return
        if force and self.ready:
            self._print('Warning: Forcing the execution of the task, even if the result is available.')
        self._print(f'The execution of the task `{self.name}` is started.')

        manager = Manager()
        collector = manager.dict()

        if not parallelize:
            cores_number = 1
        else:
            if cores_number == -1:
                cores_number = cpu_count()
            else:
                cores_number = max(cpu_count(), cores_number)

        lock = Lock()
        num_exp_per_process = round(self.n_experiments / cores_number)
        processes = [Process(target=self._serial_run, args=(n, num_exp_per_process, collector, lock)) for n
                     in range(0, cores_number)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        self._finalize_run(collector.values())
        self._finalize_execution()

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
        content = json.dumps(self.result)
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
            os.chdir('../')
        # delete the temp dir
        shutil.rmtree(temp_dir_path)
        os.chdir(cur_dir)

    def _print(self, text: str):
        if self.verbose != 0:
            print(text)

    def _finalize_execution(self):
        self.metadata['EXECUTION_DATE'] = str(datetime.datetime.now())
        self.ready = True
