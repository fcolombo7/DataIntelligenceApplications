import argparse
import os
import sys

from data_generators.basic_generator import BasicDataGenerator
from utils.tasks.task3 import Task3


def task_builder(step, source=None, simulation_name=None):
    if source is None:
        source = 'src/basic001.json'
    data_generator = BasicDataGenerator(source)
    if step == 3:
        description = 'Simulation of the step 3.'
        if simulation_name is None:
            return Task3(data_generator, description=description)
        return Task3(data_generator, name=simulation_name, description=description)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--parallelize", required=False,
                    help="Parallelization of the execution")
    ap.add_argument("--n_proc", required=False,
                    help="number of processors")
    ap.add_argument("--output_folder", required=False,
                    help="Output folder where the result of the simulation is saved")
    ap.add_argument("--simulation_name", required=False,
                    help="Name of the simulation")
    ap.add_argument("--src", required=False,
                    help="Source data")
    ap.add_argument("-s", "--step", required=True,
                    help="Step to execute")
    ap.add_argument("-T", "--time_horizon", required=True,
                    help="Time Horizon")
    ap.add_argument("-n", "--n_experiments", required=True,
                    help="Number of experiments")

    args = vars(ap.parse_args())

    task = task_builder(step=int(args['step']), source=args['src'], simulation_name=args['simulation_name'])

    task.config(int(args['time_horizon']), int(args['n_experiments']))
    parallelize = None
    if args['parallelize'] == 'true' or args['parallelize'] is None:
        parallelize = True
    else:
        parallelize = False
    n_proc = -1
    if args['n_proc'] is not None:
        n_proc = int(args['n_proc'])
    task.run(parallelize=parallelize, cores_number=n_proc)
    if args['output_folder'] is None:
        task.save()
    else:
        task.save(args['output_folder'])