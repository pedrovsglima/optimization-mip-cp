import os
import time
import numpy as np
import pandas as pd
import pulp

class SchedulingProblem:
    """A class representing a scheduling problem."""

    HEADER = ["file", "instance", "num_jobs", "num_machines", "time_limit", "result_status", "result_objective",
            "result_runtime", "upper_bound", "lower_bound", "relax_status", "relax_objective", "relax_runtime"]

    def __init__(self, data:dict, M:int=1e6):
        self.upper_bound = data["upper_bound"]
        self.lower_bound = data["lower_bound"]
        self.big_m = M

        self.num_jobs = None
        self.num_machines = None
        self.seed = None

    def _get_model(self, is_relaxed:bool=False) -> tuple:
        raise NotImplementedError("Subclasses should implement this method.")

    def solve_prob(self, time_limit:int) -> dict:
        """Solve the flow shop scheduling problem using PuLP."""

        # TODO: store optimal schedule
        # TODO: create and save gantt chart

        model, *decision_vars = self._get_model()

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, options=[f"randomSeed {self.seed}"])

        # solve model
        start_time = time.time()
        status = model.solve(solver)
        objective = pulp.value(model.objective)
        runtime = time.time() - start_time

        # LP relaxation
        model_relax, *decision_vars_relax = self._get_model(is_relaxed=True)
        solver_relax = pulp.PULP_CBC_CMD(msg=False, options=[f"randomSeed {self.seed}"])

        # solve model
        start_time_relax = time.time()
        status_relax = model_relax.solve(solver_relax)
        objective_relax = pulp.value(model_relax.objective)
        runtime_relax = time.time() - start_time_relax

        return {
            "result_status": status,
            "result_objective": objective,
            "result_runtime": runtime,
            "relax_status": status_relax,
            "relax_objective": objective_relax,
            "relax_runtime": runtime_relax,
        }

    def save_results(
            self,
            csv_path:str,
            time_limit:int,
            results:dict,
            all_info:dict
    ) -> None:
        """Save the results to a CSV file."""

        all_info["num_jobs"] = self.num_jobs
        all_info["num_machines"] = self.num_machines
        all_info["time_limit"] = time_limit
        all_info["upper_bound"] = self.upper_bound
        all_info["lower_bound"] = self.lower_bound
        all_info.update(results)

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
            self,
            is_relaxed:bool=False,
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
        if is_relaxed:
            y = pulp.LpVariable.dicts("y", [(i, i_) for i in range(self.num_jobs) for i_ in range(self.num_jobs) if i != i_], cat="Continuous", lowBound=0, upBound=1)
        else:
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
            self,
            is_relaxed:bool=False,
    ) -> tuple[
        pulp.LpProblem,
        dict[tuple[int, int], pulp.LpVariable],
        dict[tuple[int, int], pulp.LpVariable],
        pulp.LpVariable
    ]:
        """Formulate the job shop scheduling problem using PuLP."""
        model = pulp.LpProblem("JobShop", pulp.LpMinimize)

        # Decision variables        
        C = pulp.LpVariable.dicts(
            "C", 
            [(i, k) for i in range(self.num_jobs) 
                   for k in range(self.num_machines)],
            lowBound=0
        )
        # Binary precedence variables - optimized creation
        y = {}
        machine_ops = {}  # {machine: [(job, op_index)]}
        # Pre-process: Group operations by machine
        for i in range(self.num_jobs):
            for k in range(self.num_machines):
                m = self.job_operations[i][k] - 1  # Convert to 0-index
                if m not in machine_ops:
                    machine_ops[m] = []
                machine_ops[m].append((i, k))
        if is_relaxed:
            # Create only necessary precedence variables
            for m in machine_ops:
                ops = machine_ops[m]
                for idx1, (i, k) in enumerate(ops):
                    for idx2, (ip, kp) in enumerate(ops):
                        if i < ip:  # Avoid duplicates and ensure i < ip
                            y[(i, k, ip, kp)] = pulp.LpVariable(
                                f"y_{i}_{k}_{ip}_{kp}", cat="Continuous", lowBound=0, upBound=1)
        else:
            # Create only necessary precedence variables
            for m in machine_ops:
                ops = machine_ops[m]
                for idx1, (i, k) in enumerate(ops):
                    for idx2, (ip, kp) in enumerate(ops):
                        if i < ip:  # Avoid duplicates and ensure i < ip
                            y[(i, k, ip, kp)] = pulp.LpVariable(
                                f"y_{i}_{k}_{ip}_{kp}", cat="Binary")
        C_max = pulp.LpVariable("C_max", lowBound=0)

        # Objective: minimize makespan
        model += C_max

        # Constraints
        # 1. Operation precedence within jobs
        for i in range(self.num_jobs):
            for k in range(1, self.num_machines):
                prev_machine = self.job_operations[i][k-1] - 1
                curr_machine = self.job_operations[i][k] - 1
                model += C[(i, k)] >= C[(i, k-1)] + \
                    self.processing_times[i][curr_machine]

        # 2. Machine capacity constraints
        for m in machine_ops:
            ops = machine_ops[m]
            for idx1, (i, k) in enumerate(ops):
                for idx2, (ip, kp) in enumerate(ops):
                    if i < ip:  # Only compare each pair once
                        model += C[(i, k)] >= C[(ip, kp)] + \
                            self.processing_times[i][m] - \
                            self.big_m * (1 - y[(i, k, ip, kp)])
                        model += C[(ip, kp)] >= C[(i, k)] + \
                            self.processing_times[ip][m] - \
                            self.big_m * y[(i, k, ip, kp)]

        # 3. Makespan constraint
        for i in range(self.num_jobs):
            model += C_max >= C[(i, self.num_machines-1)]

        return model, C, y, C_max

class OpenShopProblem(SchedulingProblem):
    """A class representing an open shop scheduling problem."""

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
            self,
            is_relaxed:bool=False,
    ) -> tuple[
        pulp.LpProblem,
        dict[tuple[int, int], pulp.LpVariable],
        dict[tuple[int, int], pulp.LpVariable],
        pulp.LpVariable
    ]:
        """Formulate the open shop scheduling problem using PuLP."""

        model = pulp.LpProblem("OpenShop", pulp.LpMinimize)

        # Decision variables
        C = pulp.LpVariable.dicts("C", [(i, j) for i in range(self.num_jobs) 
                                  for j in range(self.num_machines)], lowBound=0)
        if is_relaxed:
            y = pulp.LpVariable.dicts("y", [(i, k, j) for j in range(self.num_machines)
                                        for i in range(self.num_jobs)
                                        for k in range(self.num_jobs)
                                        if i < k], cat="Continuous", lowBound=0, upBound=1)  # i < k prevents duplicates
            z = pulp.LpVariable.dicts("z", [(i, j, k) for i in range(self.num_jobs)
                                        for j in range(self.num_machines)
                                        for k in range(self.num_machines)
                                        if j < k], cat="Continuous", lowBound=0, upBound=1)  # j < k prevents duplicates
        else:
            y = pulp.LpVariable.dicts("y", [(i, k, j) for j in range(self.num_machines)
                                        for i in range(self.num_jobs)
                                        for k in range(self.num_jobs)
                                        if i < k], cat="Binary")  # i < k prevents duplicates
            z = pulp.LpVariable.dicts("z", [(i, j, k) for i in range(self.num_jobs)
                                        for j in range(self.num_machines)
                                        for k in range(self.num_machines)
                                        if j < k], cat="Binary")  # j < k prevents duplicates
        C_max = pulp.LpVariable("C_max", lowBound=0)

        # Objective: minimize makespan
        model += C_max

        # Constraints
        # 1. Machine capacity
        for j in range(self.num_machines):
            jobs = [i for i in range(self.num_jobs)]
            for idx_i, i in enumerate(jobs):
                for idx_k, k in enumerate(jobs):
                    if idx_i < idx_k:  # Compare each pair only once
                        model += C[(i,j)] >= C[(k,j)] + self.processing_times[i][j] - self.big_m*(1 - y[(i,k,j)])
                        model += C[(k,j)] >= C[(i,j)] + self.processing_times[k][j] - self.big_m*y[(i,k,j)]
        
        # 2. Job operation ordering
        for i in range(self.num_jobs):
            machines = [j for j in range(self.num_machines)]
            for idx_j, j in enumerate(machines):
                for idx_k, k in enumerate(machines):
                    if idx_j < idx_k:  # Compare each pair only once
                        model += C[(i,j)] >= C[(i,k)] + self.processing_times[i][j] - self.big_m*(1 - z[(i,j,k)])
                        model += C[(i,k)] >= C[(i,j)] + self.processing_times[i][k] - self.big_m*z[(i,j,k)]
        
        # 3. Makespan
        for i in range(self.num_jobs):
            for j in range(self.num_machines):
                model += C_max >= C[(i,j)]

        return model, C, y, z, C_max
