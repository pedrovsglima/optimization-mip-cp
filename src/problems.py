import os
import time
import numpy as np
import pandas as pd
import pulp

class SchedulingProblem:
    """A class representing a scheduling problem."""

    HEADER = ["file", "instance", "num_jobs", "num_machines", "time_limit",
            "result_status", "result_objective", "result_runtime", "upper_bound", "lower_bound"]

    def __init__(self, data:dict, M:int=1e6):
        self.upper_bound = data["upper_bound"]
        self.lower_bound = data["lower_bound"]
        self.big_m = M

        self.num_jobs = None
        self.num_machines = None
        self.seed = None

    def _get_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def solve_prob(self, time_limit:int) -> tuple[int, int, float]:
        """Solve the flow shop scheduling problem using PuLP."""

        # TODO: store optimal schedule
        # TODO: create and save gantt chart

        model, C, y, C_max = self._get_model()

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, options=[f"randomSeed {self.seed}"])

        # solve model
        start_time = time.time()
        status = model.solve(solver)
        objective = pulp.value(model.objective)
        runtime = time.time() - start_time

        return status, objective, runtime

    def save_results(
            self,
            csv_path:str,
            time_limit:int,
            status:int,
            objective:int,
            runtime:float,
            all_info:dict
    ) -> None:
        """Save the results to a CSV file."""

        all_info["num_jobs"] = self.num_jobs
        all_info["num_machines"] = self.num_machines
        all_info["time_limit"] = time_limit
        all_info["result_status"] = status
        all_info["result_objective"] = objective
        all_info["result_runtime"] = runtime
        all_info["upper_bound"] = self.upper_bound
        all_info["lower_bound"] = self.lower_bound

        df = pd.DataFrame([all_info], columns=self.HEADER)

        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode="a", header=False, index=False)

class FlowShopProblem(SchedulingProblem):
    """A class representing a flow shop scheduling problem."""

    def __init__(self, data:dict):
        super().__init__(data)
        self.num_jobs = data["nb_jobs"]
        self.num_machines = data["nb_machines"]
        self.processing_times = np.array(data["times"]).T
        self.seed = str(data["seed"])

        if self.processing_times.shape != (self.num_jobs, self.num_machines):
            raise ValueError("The dimensions of processing_times do not match num_jobs and num_machines.")

    def _get_model(
            self
    ) -> tuple[
        pulp.LpProblem,
        dict[tuple[int, int], pulp.LpVariable],
        dict[tuple[int, int], pulp.LpVariable],
        pulp.LpVariable
    ]:
        """Formulate the flow shop scheduling problem using PuLP."""

        model = pulp.LpProblem("FlowShop", pulp.LpMinimize)

        # Decision variables
        C = pulp.LpVariable.dicts("C", [(i, j) for i in range(self.num_jobs) for j in range(self.num_machines)], lowBound=0)
        y = pulp.LpVariable.dicts("y", [(i, i_) for i in range(self.num_jobs) for i_ in range(self.num_jobs) if i != i_], cat="Binary")
        C_max = pulp.LpVariable("C_max", lowBound=0)

        # Objective: minimize makespan
        model += C_max

        # Constraints
        for i in range(self.num_jobs):
            model += C[i, 0] >= self.processing_times[i][0]  # First machine processing constraint
            for j in range(1, self.num_machines):
                model += C[i, j] >= C[i, j-1] + self.processing_times[i][j]  # Sequential processing on machines

        # No overlapping constraint (job ordering)
        for i in range(self.num_jobs):
            for i_ in range(self.num_jobs):
                if i != i_:
                    for j in range(self.num_machines):
                        model += C[i, j] >= C[i_, j] + self.processing_times[i][j] - self.big_m * (1 - y[i, i_])
                        model += C[i_, j] >= C[i, j] + self.processing_times[i_][j] - self.big_m * y[i, i_]

        # Makespan constraint
        for i in range(self.num_jobs):
            model += C_max >= C[i, self.num_machines - 1]

        return model, C, y, C_max

class JobShopProblem(SchedulingProblem):
    """A class representing a job shop scheduling problem."""

    def __init__(self, data:dict):
        super().__init__(data)
        self.num_jobs = data["nb_jobs"]
        self.num_machines = data["nb_machines"]
        self.processing_times = data["times"]
        self.job_operations = data["machines"]
        self.seed = str(data["time_seed"])
        self.machine_seed = str(data["machine_seed"])

        if len(self.processing_times) != self.num_jobs or len(self.processing_times[0]) != self.num_machines:
            raise ValueError("The dimensions of processing_times do not match num_jobs and num_machines.")
        if len(self.job_operations) != self.num_jobs or len(self.job_operations[0]) != self.num_machines:
            raise ValueError("The dimensions of job_operations do not match num_jobs and num_machines.")

    def _get_model(
            self
    ) -> tuple[
        pulp.LpProblem,
        dict[tuple[int, int], pulp.LpVariable],
        dict[tuple[int, int], pulp.LpVariable],
        pulp.LpVariable
    ]:
        """Formulate the job shop scheduling problem using PuLP."""
        model = pulp.LpProblem("JobShop", pulp.LpMinimize)

        # Decision variables
        C = pulp.LpVariable.dicts("C", [(i, j) for i in range(self.num_jobs) for j in range(self.num_machines)], lowBound=0)
        y = pulp.LpVariable.dicts("y", [(i, i_) for i in range(self.num_jobs) for i_ in range(self.num_jobs) if i != i_], cat="Binary")
        C_max = pulp.LpVariable("C_max", lowBound=0)

        # Objective: minimize makespan
        model += C_max

        # Constraints
        # Sequence constraints within a job
        for i in range(self.num_jobs):
            for k in range(1, self.num_machines):
                prev = self.job_operations[i][k - 1] - 1  # -1 for 0-indexing
                curr = self.job_operations[i][k] - 1  # -1 for 0-indexing
                model += C[i, curr] >= C[i, prev] + self.processing_times[i][prev]

        # No overlapping constraint (job ordering)
        for j in range(self.num_machines):
            for i in range(self.num_jobs):
                for i_ in range(self.num_jobs):
                    if i != i_ and self.job_operations[i].count(j+1) and self.job_operations[i_].count(j+1):
                        model += C[i, j] >= C[i_, j] + self.processing_times[i][j] - self.big_m * (1 - y[i, i_])
                        model += C[i_, j] >= C[i, j] + self.processing_times[i_][j] - self.big_m * y[i, i_]

        # Makespan constraint
        for i in range(self.num_jobs):
            last_machine = self.job_operations[i][-1] - 1  # -1 for 0-indexing
            model += C_max >= C[i, last_machine]

        return model, C, y, C_max
