import os
import time
import numpy as np
import pandas as pd

from docplex.cp.config import context
from docplex.cp.model import (
    CpoModel, interval_var, sequence_var, end_of, no_overlap, start_of,
    max as cp_max, minimize, first, same_sequence, end_before_start
)


class SchedulingProblem:
    """A class representing a scheduling problem."""

    HEADER = ["file", "instance", "num_jobs", "num_machines", "solver", "time_limit",
            "result_status", "result_objective", "result_runtime", "upper_bound", "lower_bound"]

    def __init__(self, data:dict):
        self.upper_bound = data["upper_bound"]
        self.lower_bound = data["lower_bound"]

        self.num_jobs = None
        self.num_machines = None
        self.seed = None

    def _get_model(self) -> CpoModel:
        raise NotImplementedError("Subclasses should implement this method.")

    def solve_prob(self, time_limit:int, header:dict) -> dict:
        """Solve the flow shop scheduling problem using PuLP."""

        model = self._get_model()

        context.solver.agent = header["cpoptimizer_agent"]
        context.solver.local.execfile = header["cpoptimizer_path"]

        # solve model
        start_time = time.time()
        solution = model.solve(TimeLimit=time_limit, LogVerbosity="Quiet", RandomSeed=self.seed)
        runtime = time.time() - start_time

        return {
            "result_status": 1 if solution.is_solution() else 0,
            "result_objective": solution.get_objective_value(),
            "result_runtime": runtime,
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
        self.seed = int(data["seed"])

        if self.processing_times.shape != (self.num_jobs, self.num_machines):
            raise ValueError("The dimensions of processing_times do not match num_jobs and num_machines.")

    def _get_model(self) -> CpoModel:
        """Formulate the flow shop scheduling problem using DOcplex."""
        mdl = CpoModel()

        processing_times = self.processing_times.astype(int)
        upper_bound = int(np.sum(np.max(processing_times, axis=0)))

        # interval variables for each operation of each job
        operations = []
        for j in range(self.num_jobs):
            job_ops = []
            for m in range(self.num_machines):
                dur = processing_times[j, m]
                op = interval_var(size=dur, end=(0, upper_bound), name=f"O_{j}_{m}")
                job_ops.append(op)
            operations.append(job_ops)

        # sequence variable for the first machine
        first_machine_ops = [operations[j][0] for j in range(self.num_jobs)]
        job_order = sequence_var(first_machine_ops, name="job_order")

        # fix job 0 as first (symmetry breaking)
        mdl.add(first(job_order, operations[0][0]))

        # sequence variables for all other machines
        machine_sequences = []
        for m in range(self.num_machines):
            machine_ops = [operations[j][m] for j in range(self.num_jobs)]
            seq_var = sequence_var(machine_ops, name=f"machine_seq_{m}")
            machine_sequences.append(seq_var)

        # all machines with the same job order as the first machine
        for m in range(1, self.num_machines):
            mdl.add(same_sequence(job_order, machine_sequences[m]))

        # operations within a job must follow machine order
        for j in range(self.num_jobs):
            for m in range(self.num_machines - 1):
                mdl.add(end_before_start(operations[j][m], operations[j][m + 1]))

        # makespan
        last_ops = [operations[j][self.num_machines - 1] for j in range(self.num_jobs)]
        makespan = cp_max([end_of(op) for op in last_ops])
        mdl.add(makespan <= upper_bound)

        mdl.add(minimize(makespan))

        return mdl

class JobShopProblem(SchedulingProblem):
    """A class representing a job shop scheduling problem."""

    def __init__(self, data:dict):
        super().__init__(data)
        self.num_jobs = data["nb_jobs"]
        self.num_machines = data["nb_machines"]
        self.processing_times = np.array(data["times"])
        self.job_operations = data["machines"]
        self.seed = int(data["time_seed"])
        self.machine_seed = int(data["machine_seed"])

        if len(self.processing_times) != self.num_jobs or len(self.processing_times[0]) != self.num_machines:
            raise ValueError("The dimensions of processing_times do not match num_jobs and num_machines.")
        if len(self.job_operations) != self.num_jobs or len(self.job_operations[0]) != self.num_machines:
            raise ValueError("The dimensions of job_operations do not match num_jobs and num_machines.")

    def _get_model(self) -> CpoModel:
        """Formulate the job shop scheduling problem using DOcplex."""
        mdl = CpoModel()

        processing_times = self.processing_times.astype(int)
        upper_bound = int(np.sum(processing_times))

        # interval variables for each operation of each job
        operations = []
        for j in range(self.num_jobs):
            job_ops = []
            for k in range(self.num_machines):
                m = self.job_operations[j][k] - 1  # 0-index
                dur = processing_times[j][m]
                op = interval_var(size=dur, end=(0, upper_bound), name=f"O_{j}_{k}_M{m}")
                job_ops.append(op)
            operations.append(job_ops)

        # precedence constraints
        for j in range(self.num_jobs):
            for k in range(self.num_machines - 1):
                mdl.add(end_before_start(operations[j][k], operations[j][k + 1]))

        # machine capacity constraints
        for m in range(self.num_machines):
            machine_ops = []
            for j in range(self.num_jobs):
                for k in range(self.num_machines):
                    if self.job_operations[j][k] - 1 == m:
                        machine_ops.append(operations[j][k])
            if machine_ops:
                mdl.add(no_overlap(machine_ops))

        # makespan
        last_ops = [operations[j][-1] for j in range(self.num_jobs)]
        makespan = cp_max([end_of(op) for op in last_ops])
        mdl.add(makespan <= upper_bound)
        mdl.add(minimize(makespan))

        return mdl

class OpenShopProblem(SchedulingProblem):
    """A class representing an open shop scheduling problem."""

    def __init__(self, data:dict):
        super().__init__(data)
        self.num_jobs = data["nb_jobs"]
        self.num_machines = data["nb_machines"]
        self.processing_times = np.array(data["times"])
        self.job_operations = data["machines"]
        self.seed = int(data["time_seed"])
        self.machine_seed = int(data["machine_seed"])

        if len(self.processing_times) != self.num_jobs or len(self.processing_times[0]) != self.num_machines:
            raise ValueError("The dimensions of processing_times do not match num_jobs and num_machines.")
        if len(self.job_operations) != self.num_jobs or len(self.job_operations[0]) != self.num_machines:
            raise ValueError("The dimensions of job_operations do not match num_jobs and num_machines.")

    def _get_model(self) -> CpoModel:
        """Formulate the open shop scheduling problem using DOcplex."""
        mdl = CpoModel()

        processing_times = self.processing_times.astype(int)
        upper_bound = int(np.sum(processing_times))

        # interval variables for each operation (job i on machine j)
        operations = []
        for i in range(self.num_jobs):
            job_ops = []
            for j in range(self.num_machines):
                dur = processing_times[i, j]
                op = interval_var(size=dur, end=(0, upper_bound), name=f"O_{i}_{j}")
                job_ops.append(op)
            operations.append(job_ops)

        # machine capacity constraints
        for j in range(self.num_machines):
            machine_ops = [operations[i][j] for i in range(self.num_jobs)]
            mdl.add(no_overlap(machine_ops))

        # job capacity constraints
        for i in range(self.num_jobs):
            job_ops = operations[i]
            mdl.add(no_overlap(job_ops))

        # fix job 0 as first (symmetry breaking)
        mdl.add(start_of(operations[0][0]) == 0)

        # makespan
        all_ends = [end_of(operations[i][j]) for i in range(self.num_jobs) for j in range(self.num_machines)]
        makespan = cp_max(all_ends)
        mdl.add(minimize(makespan))

        return mdl
