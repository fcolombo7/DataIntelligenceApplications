import argparse
from learners.GPTS_Learner_v3 import GPTS_Learner
from learners.joint_contextual_learner import JointContextualLearner
from learners.joint_learner_v2 import JointLearner
from learners.pricing.thompson_sampling import ThompsonSampling
from learners.pricing.ucb import UCB
from utils.tasks.complete_task import CompleteTask


def task_builder(simulation_name, source, fixed_adv, fixed_price, selected_bid, selected_price, pricing_context,
                 time_horizon, n_experiments, cg_confidence, cg_start_from, cg_frequency):
    new_task = CompleteTask(data_src=source,
                            name=simulation_name,
                            fixed_adv=fixed_adv,
                            fixed_price=fixed_price,
                            selected_bid=selected_bid,
                            selected_price=selected_price,
                            pricing_context=pricing_context
                            )
    if fixed_price:
        learners = [GPTS_Learner]
    elif fixed_adv:
        learners = [UCB, ThompsonSampling]
    elif pricing_context:
        learners = [JointContextualLearner]
    else:
        learners = [JointLearner]

    new_task.config(time_horizon=time_horizon,
                    n_experiments=n_experiments,
                    learner_to_test=learners,
                    cg_start_from=cg_start_from,
                    cg_frequency=cg_frequency,
                    cg_confidence=cg_confidence)
    return new_task


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--simulation_name", required=True, help="Name of the simulation", type=str)
    ap.add_argument("--src", required=True, help="Source data", type=str)
    ap.add_argument("-T", "--time_horizon", required=True, help="Time Horizon", type=int)
    ap.add_argument("-n", "--n_experiments", required=True, help="Number of experiments", type=int)

    ap.add_argument("--fixed_adv", required=False, help="Fixed advertisement campaign",
                    type=bool, default=False, choices=[True, False])
    ap.add_argument("--fixed_price", required=False, help="Fixed pricing campaign",
                    type=bool, default=False, choices=[True, False])
    ap.add_argument("--selected_bid", required=False, help="Selected bid (in case of fixed_adv)",
                    type=int, default=4, choices=range(0, 10))
    ap.add_argument("--selected_price", required=False, help="Selected price (in case of fixed_price)",
                    type=int, default=3, choices=range(0, 10))
    ap.add_argument("--pricing_context", required=False,
                    help="Contextual pricing campaign (in case of not fixed_price)",
                    type=bool, default=False, choices=[True, False])

    ap.add_argument("--cg_start_from", required=False, help="Context generator starting day",
                    type=int, default=31)
    ap.add_argument("--cg_frequency", required=False, help="Context generator frequency",
                    type=int, default=10)
    ap.add_argument("--cg_confidence", required=False,
                    help="Context generator confidence",
                    type=float, default=0.002)

    ap.add_argument("--output_folder", required=False, help="Output folder where the result of the simulation is saved",
                    type=str, default='simulations_results')

    args = vars(ap.parse_args())
    print(args)
    task = task_builder(source=args['src'],
                        simulation_name=args['simulation_name'],
                        fixed_adv=args['fixed_adv'],
                        fixed_price=args['fixed_price'],
                        selected_bid=args['selected_bid'],
                        selected_price=args['selected_price'],
                        pricing_context=args['pricing_context'],
                        time_horizon=args['time_horizon'],
                        n_experiments=args['n_experiments'],
                        cg_frequency=args['cg_frequency'],
                        cg_confidence=args['cg_confidence'],
                        cg_start_from=args['cg_start_from']
                        )

    out_folder = args['output_folder']
    task.run()
    path = task.save(folder=out_folder)
    task.load(path)
    task.plot(0)
    task.plot(1)
    task.plot(2)