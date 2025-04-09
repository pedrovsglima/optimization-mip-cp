import os
import toml
import logging
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

    image_dir = config["data"]["directory"]

    # open shop
    # file_path = "data/openshop_tai4_4.txt"
    # open_shop = read_instances.OpenShop(file_path)
    # print(open_shop.data[2])

    # flow shop
    flowshop_prefix = config["data"]["flow_prefix"]
    flowshop_files = [os.path.join(image_dir, f)
                      for f in os.listdir(image_dir) if flowshop_prefix in f]

    for file_path in flowshop_files:
        fs_instances = read_instances.FlowShop(file_path)
        for instance, data in enumerate(fs_instances.data):
            for time_limit in config["optimization"]["time_limit"]:

                save_info = {"file": file_path, "instance": instance}
                logger.info("-----------------------------------------")
                logger.info("File: %s, Instance: %s", file_path, instance)

                fs = problems.FlowShopProblem(data)
                status, objective, runtime = fs.solve_prob(time_limit)
                logger.info("Status: %s, Objective: %s, Runtime: %s", status, objective, runtime)

                fs.save_results(config["results"]["csv_path"], time_limit, status, objective, runtime, save_info)

    # job shop
    jobshop_prefix = config["data"]["job_prefix"]
    jobshop_files = [os.path.join(image_dir, f)
                     for f in os.listdir(image_dir) if jobshop_prefix in f]

    for file_path in jobshop_files:
        js_instances = read_instances.JobShop(file_path)
        for instance, data in enumerate(js_instances.data):
            for time_limit in config["optimization"]["time_limit"]:

                save_info = {"file": file_path, "instance": instance}
                logger.info("-----------------------------------------")
                logger.info("File: %s, Instance: %s", file_path, instance)

                js = problems.JobShopProblem(data)
                status, objective, runtime = js.solve_prob(time_limit)
                logger.info("Status: %s, Objective: %s, Runtime: %s", status, objective, runtime)

                js.save_results(config["results"]["csv_path"], time_limit, status, objective, runtime, save_info)
