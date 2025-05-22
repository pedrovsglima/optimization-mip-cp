import os
import toml
import random
import logging
import argparse
from src import read_instances, problems

# set up logging
logging.basicConfig(
    filename="./log/experiments.log", level=logging.INFO,
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

# config file
with open("config.toml", "r") as f:
    config = toml.load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", type=str, choices=["all", "flowshop", "jobshop", "openshop"], default="all")
    args = parser.parse_args()

    prob = args.prob
    N_INSTANCES = 5

    image_dir = config["data"]["directory"]

    # flow shop
    if prob in ["all", "flowshop"]:
        flowshop_prefix = config["data"]["flow_prefix"]
        flowshop_files = [os.path.join(image_dir, f)
                        for f in os.listdir(image_dir) if flowshop_prefix in f]
        flowshop_files = random.sample(flowshop_files, k=min(N_INSTANCES, len(flowshop_files)))

        for file_path in flowshop_files:
            fs_instances = read_instances.FlowShop(file_path)
            for instance, data in enumerate(fs_instances.data):
                for time_limit in config["optimization"]["time_limit"]:

                    save_info = {"file": file_path, "instance": instance, "solver": config["optimization"]["solver"]}
                    logger.info("-----------------------------------------")
                    logger.info("File: %s, Instance: %s", file_path, instance)

                    fs = problems.FlowShopProblem(data)
                    results = fs.solve_prob(time_limit, config["optimization"])
                    logger.info("Status: %s, Objective: %s, Runtime: %s",
                                results["result_status"],
                                results["result_objective"],
                                results["result_runtime"],
                    )
                    fs.save_results(config["results"]["csv_path"], time_limit, results, save_info)

    # job shop
    if prob in ["all", "jobshop"]:
        jobshop_prefix = config["data"]["job_prefix"]
        jobshop_files = [os.path.join(image_dir, f)
                        for f in os.listdir(image_dir) if jobshop_prefix in f]
        jobshop_files = random.sample(jobshop_files, k=min(N_INSTANCES, len(jobshop_files)))

        for file_path in jobshop_files:
            js_instances = read_instances.JobShop(file_path)
            for instance, data in enumerate(js_instances.data):
                for time_limit in config["optimization"]["time_limit"]:

                    save_info = {"file": file_path, "instance": instance, "solver": config["optimization"]["solver"]}
                    logger.info("-----------------------------------------")
                    logger.info("File: %s, Instance: %s", file_path, instance)

                    js = problems.JobShopProblem(data)
                    results = js.solve_prob(time_limit, config["optimization"])
                    logger.info("Status: %s, Objective: %s, Runtime: %s",
                                results["result_status"],
                                results["result_objective"],
                                results["result_runtime"],
                    )
                    js.save_results(config["results"]["csv_path"], time_limit, results, save_info)

    # open shop
    if prob in ["all", "openshop"]:
        openshop_prefix = config["data"]["open_prefix"]
        openshop_files = [os.path.join(image_dir, f)
                        for f in os.listdir(image_dir) if openshop_prefix in f]
        openshop_files = random.sample(openshop_files, k=min(N_INSTANCES, len(openshop_files)))

        for file_path in openshop_files:
            os_instances = read_instances.OpenShop(file_path)
            for instance, data in enumerate(os_instances.data):
                for time_limit in config["optimization"]["time_limit"]:

                    save_info = {"file": file_path, "instance": instance, "solver": config["optimization"]["solver"]}
                    logger.info("-----------------------------------------")
                    logger.info("File: %s, Instance: %s", file_path, instance)

                    opens = problems.OpenShopProblem(data)
                    results = opens.solve_prob(time_limit, config["optimization"])
                    logger.info("Status: %s, Objective: %s, Runtime: %s",
                                results["result_status"],
                                results["result_objective"],
                                results["result_runtime"],
                    )
                    opens.save_results(config["results"]["csv_path"], time_limit, results, save_info)
