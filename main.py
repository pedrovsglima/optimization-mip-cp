import os
import time
import logging
import numpy as np
import pandas as pd
import toml
import pulp

from src.data import read_instances
from src.models import flowshop


# set up logging
logging.basicConfig(
    filename="./log/experiments.log", level=logging.INFO,
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

# config file
with open("config.toml", "r") as f:
    config = toml.load(f)


def main(input_data:dict, time_limit:int, all_info:dict) -> None:

    # TODO: store optimal schedule
    # TODO: create and save gantt chart

    num_jobs = input_data["nb_jobs"]
    num_machines = input_data["nb_machines"]
    processing_times = np.array(input_data["times"]).T
    seed = str(input_data["seed"])

    logger.info("Number of jobs: %d, Number of machines: %d, Time limit: %d s",
                num_jobs, num_machines, time_limit)

    # create model and solver
    model, C, y, C_max = flowshop.model(num_jobs, num_machines, processing_times)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, options=[f"randomSeed {seed}"])

    # solve model
    start_time = time.time()

    status = model.solve(solver)
    objective = pulp.value(model.objective)

    runtime = time.time() - start_time

    # log results
    logger.info("Status: %s, Objective: %d, Runtime: %.2f s", status, objective, runtime)

    # compare with best known solution
    upper_bound = input_data["upper_bound"]
    lower_bound = input_data["lower_bound"]
    logger.info("Known Upper Bound: %d, Known Lower Bound: %d", upper_bound, lower_bound)

    if status == 0:
        logger.info("Could not find an optimal solution within the time limit")
    else:
        if objective <= lower_bound:
            logger.info("Optimal solution found")
        elif lower_bound < objective <= upper_bound:
            logger.info("Feasible solution: %d, but may be improved", objective)
        else:
            logger.info("Solution %d is above the known upper bound", objective)

        gap = ((objective - lower_bound) / (upper_bound - lower_bound)) * 100
        logger.info("Performance Gap: %.2f%%", gap)

    # info to csv
    csv_path = config["results"]["csv_path"]

    all_info["num_jobs"] = num_jobs
    all_info["num_machines"] = num_machines
    all_info["time_limit"] = time_limit
    all_info["result_status"] = status
    all_info["result_objective"] = objective
    all_info["result_runtime"] = runtime
    all_info["upper_bound"] = upper_bound
    all_info["lower_bound"] = lower_bound

    df = pd.DataFrame([all_info])
    df = df[["file", "instance", "num_jobs", "num_machines", "time_limit",
             "result_status", "result_objective", "result_runtime", "upper_bound", "lower_bound"]]

    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


if __name__ == "__main__":

    image_dir = config["data"]["directory"]

    # job shop
    # file_path = "data/jobshop_tai15_15.txt"
    # jobshop = read_instances.JobShop(file_path)
    # print(jobshop.data[1])

    # open shop
    # file_path = "data/openshop_tai4_4.txt"
    # open_shop = read_instances.OpenShop(file_path)
    # print(open_shop.data[2])

    # flow shop
    flowshop_prefix = config["data"]["flow_prefix"]
    flowshop_files = [os.path.join(image_dir, f) 
                      for f in os.listdir(image_dir) if flowshop_prefix in f]

    for time_limit in config["optimization"]["time_limit"]:
        for file_path in flowshop_files:
            fs = read_instances.FlowShop(file_path)

            for instance, data in enumerate(fs.data):
                save_info = {"file": file_path, "instance": instance}
                logger.info("-----------------------------------------")
                logger.info("File: %s, Instance: %s", file_path, instance)

                main(data, time_limit, save_info)

                break
            break
        break
