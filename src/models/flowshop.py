import numpy as np
import pulp

def model(num_jobs: int, num_machines: int, processing_times: np.ndarray):

    if processing_times.shape != (num_jobs, num_machines):
        raise ValueError("The dimensions of the processing_times matrix do not match the number of jobs and machines.")

    model = pulp.LpProblem("FlowShop", pulp.LpMinimize)

    # Decision Variables
    C = [[pulp.LpVariable(f"C_{i}_{j}", lowBound=0) for j in range(num_machines)] for i in range(num_jobs)]
    x = [[pulp.LpVariable(f"x_{i}_{k}", cat="Binary") for k in range(num_jobs)] for i in range(num_jobs)]
    C_max = pulp.LpVariable("C_max", lowBound=0)

    # Objective: Minimize makespan
    model += C_max

    # Constraints
    # 1. Each job is assigned exactly once
    for i in range(num_jobs):
        model += pulp.lpSum(x[i][k] for k in range(num_jobs)) == 1

    # 2. Each position gets only one job
    for k in range(num_jobs):
        model += pulp.lpSum(x[i][k] for i in range(num_jobs)) == 1

    # 3. First machine processing constraint
    for i in range(num_jobs):
        # model += C[i][0] >= processing_times[i][0]
        model += C[i][0] >= pulp.lpSum(x[i][k] * processing_times[i][0] for k in range(num_jobs))

    # 4. Sequential job execution on the same machine
    for j in range(1, num_machines):
        for i in range(num_jobs):
            # model += C[i][j] >= C[i][j - 1] + processing_times[i][j]
            model += C[i][j] >= C[i][j - 1] + pulp.lpSum(x[i][k] * processing_times[i][j] for k in range(num_jobs))

    # 5. Job precedence (No overlapping jobs on the same machine)
    M = 1e6  # Large constant for Big-M constraint
    for i in range(1, num_jobs):
        for k in range(num_jobs):
            for j in range(num_machines):
                model += C[i][j] >= C[i-1][j] + processing_times[i][j] - M * (1 - x[i][k])

    # 6. Makespan constraint
    for i in range(num_jobs):
        model += C_max >= C[i][num_machines - 1]

    return model, C, x, C_max
