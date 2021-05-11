import datetime
import json
import os
import shutil
from typing import Dict
from abc import ABC, abstractmethod
from multiprocessing import Process, cpu_count, Manager
from zipfile import ZipFile


class Task(ABC):
    """
    Abstract class used to represent the tasks run by the simulator
    """
    def __init__(self, name: str, description: str, verbose: int):
        """
        Class constructor
        :param name: the name of the task
        :param description: the description of the task
        :param verbose: level of verbose. (0 or 1)
        """
        self.name = name
        self.description = description
        self.verbose = verbose
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
    def _serial_run(self, process_id: int, n_experiments: int, collector: Dict) -> None:
        """
        Single core execution of the simulation.
        :param process_id: identifier of the process.
        :param n_experiments: number of experiments tried by the single core.
        :param collector: dictionary used as shared state between cores.
        Each core should save values in collector[process_id].
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
    def _finalize_run(self, collected_values: list) -> None:
        """
        Method used to aggregate all the result computed by each core, and set the final result [`result`: dict]
        :param collected_values: values computed by each core
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

    def run(self, force=False, parallelize=True, cores_number=-1) -> None:
        """
        Method used to start the simulation of the Task.
        :param force: if False (default) check if the task has been already executed/loaded. If so the execution is skipped.
        if True, the execution is performed in any case.
        :param parallelize: if True the simulation is run in parallel, otherwise serially.
        :param cores_number: if parallelize = True, it represents the number of cores used by the simulator.
        (default= -1 =>it uses all the available cores)
        """
        # check if the config method has been called TODO: check when other tasks are available.
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

        num_exp_per_process = round(self.n_experiments / cores_number)
        processes = [Process(target=self._serial_run, args=(n, num_exp_per_process, collector)) for n
                     in range(0, cores_number)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        self._finalize_run(collector.values())
        self._finalize_execution()

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

    def save(self, folder='simulations_results', overwrite=False) -> None:
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
