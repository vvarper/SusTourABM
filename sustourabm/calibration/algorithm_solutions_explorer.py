def get_best_solution(algorithm, problem_name):
    # Print results ###########################################################

    front = algorithm.get_result()

    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem_name)
    print("Computing time: " + str(algorithm.total_computing_time))

    if type(front) is not list:
        front = [front]

    for (i, solution) in enumerate(front):
        print(f"Solution {i}: Score {solution.objectives[0]}, variables: "
              f"{solution.variables}")

    # Find solution with best score ###########################################

    best_solution = front[0]
    best_score = best_solution.objectives[0]

    for solution in front:
        if solution.objectives[0] < best_score:
            best_score = solution.objectives[0]
            best_solution = solution

    print(f"\nBest solution: Score {best_score}, variables: "
          f"{best_solution.variables}")

    return best_solution
