
from src.data import read_instances


def main():

    # flow shop
    file_path = "data/flowshop_tai20_5.txt"
    flowshop = read_instances.FlowShop(file_path)
    print(flowshop.data[5])

    # job shop
    file_path = "data/jobshop_tai15_15.txt"
    jobshop = read_instances.JobShop(file_path)
    print(jobshop.data[1])

    # open shop
    file_path = "data/openshop_tai4_4.txt"
    open_shop = read_instances.OpenShop(file_path)
    print(open_shop.data[2])


if __name__ == '__main__':
    main()
