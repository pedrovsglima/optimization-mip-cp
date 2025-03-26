import numpy as np
import pulp

def model(num_jobs: int, num_machines: int, processing_times: np.ndarray):

    if processing_times.shape != (num_jobs, num_machines):
        raise ValueError("The dimensions of the processing_times matrix do not match the number of jobs and machines.")

    model = pulp.LpProblem("FlowShop", pulp.LpMinimize)

    # Decision Variables
    C = pulp.LpVariable.dicts("C", [(i, j) for i in range(num_jobs) for j in range(num_machines)], lowBound=0, cat="Continuous")
    y = pulp.LpVariable.dicts("y", [(i, i_) for i in range(num_jobs) for i_ in range(num_jobs) if i != i_], cat="Binary")
    C_max = pulp.LpVariable("C_max", lowBound=0, cat="Continuous")

    # Objective: Minimize makespan
    model += C_max

    # Constraints
    # 1. First machine processing constraint
    for i in range(num_jobs):
        model += C[i, 0] >= processing_times[i][0]

    # 2. Sequential processing on machines
    for i in range(num_jobs):
        for j in range(1, num_machines):
            model += C[i, j] >= C[i, j - 1] + processing_times[i][j]

    # 3. No Overlapping Constraint (Job Ordering)
    M = 1e6  # Large constant for Big-M constraint
    for i in range(num_jobs):
        for i_ in range(num_jobs):
            if i != i_:
                for j in range(num_machines):
                    model += C[i, j] >= C[i_, j] + processing_times[i][j] - M * (1 - y[i, i_])
                    model += C[i_, j] >= C[i, j] + processing_times[i_][j] - M * y[i, i_]

    # 4. Consistency in Job Precedence (Transitivity)
    # TODO: também testar sem essa restrição
    for i in range(num_jobs):
        for i_ in range(num_jobs):
            for i__ in range(num_jobs):
                if i != i_ and i_ != i__ and i != i__:
                    model += y[i, i__] >= y[i, i_] + y[i_, i__] - 1

    # 5. Makespan constraint
    for i in range(num_jobs):
        model += C_max >= C[i, num_machines - 1]

    return model, C, y, C_max
