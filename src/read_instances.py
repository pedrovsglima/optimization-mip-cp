import re


class ReadData:
    HEADER = ""

    def __init__(self, filename):
        self.filename = filename
        self.instances = self.read_data()

    def read_data(self):
        with open(self.filename, "r") as file:
            return file.read()

    def process_data(self) -> dict:
        raise NotImplementedError("Subclass must implement abstract method")


class FlowShop(ReadData):
    HEADER = "number of jobs, number of machines, initial seed, upper bound and lower bound :"

    def __init__(self, filename):
        super().__init__(filename)
        self.data = self.process_data()

    def process_data(self) -> dict:
        instances = []
        raw_instances = self.read_data().strip().split(self.HEADER)

        for instance in raw_instances[1:]:
            lines = instance.strip().split("\n")
            header_values = list(map(int, re.findall(r"\d+", lines[0])))
            num_jobs, num_machines, seed, upper_bound, lower_bound = header_values

            processing_times = []
            for line in lines[2:]:  # skip the "processing times :" line
                processing_times.append(list(map(int, line.split())))

            instances.append({
                "nb_jobs": num_jobs,
                "nb_machines": num_machines,
                "seed": seed,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "times": processing_times
            })

        return instances


class JobShop(ReadData):
    HEADER = "Nb of jobs, Nb of Machines, Time seed, Machine seed, Upper bound, Lower bound"

    def __init__(self, filename):
        super().__init__(filename)
        self.data = self.process_data()

    def process_data(self) -> dict:
        instances = []
        raw_instances = re.split(self.HEADER, self.read_data())

        for raw_instance in raw_instances[1:]:
            lines = raw_instance.strip().split("\n")
            metadata = list(map(int, re.findall(r"\d+", lines[0])))
            nb_jobs, nb_machines, time_seed, machine_seed, upper_bound, lower_bound = metadata

            times_start_idx = lines.index("Times") + 1
            machines_start_idx = lines.index("Machines") + 1

            times = []
            for line in lines[times_start_idx:machines_start_idx - 1]:
                times.append(list(map(int, re.findall(r'\d+', line))))

            machines = []
            for line in lines[machines_start_idx:]:
                machines.append(list(map(int, re.findall(r'\d+', line))))

            instances.append({
                "nb_jobs": nb_jobs,
                "nb_machines": nb_machines,
                "time_seed": time_seed,
                "machine_seed": machine_seed,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "times": times,
                "machines": machines
            })

        return instances


class OpenShop(ReadData):
    HEADER = "number of jobs, number of machines, time seed, machine seed, upper bound, lower bound :"

    def __init__(self, filename):
        super().__init__(filename)
        self.data = self.process_data()

    def process_data(self) -> dict:
        instances = []
        raw_instances = re.split(self.HEADER, self.read_data(), flags=re.IGNORECASE)

        for raw_instance in raw_instances[1:]:
            lines = raw_instance.strip().split("\n")
            metadata = list(map(int, re.findall(r"\d+", lines[0])))
            nb_jobs, nb_machines, time_seed, machine_seed, upper_bound, lower_bound = metadata

            times_start_idx = lines.index("processing times :") + 1
            machines_start_idx = lines.index("machines :") + 1

            times = []
            for line in lines[times_start_idx:machines_start_idx - 1]:
                times.append(list(map(int, re.findall(r'\d+', line))))

            machines = []
            for line in lines[machines_start_idx:]:
                machines.append(list(map(int, re.findall(r'\d+', line))))

            instances.append({
                "nb_jobs": nb_jobs,
                "nb_machines": nb_machines,
                "time_seed": time_seed,
                "machine_seed": machine_seed,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "times": times,
                "machines": machines
            })

        return instances
