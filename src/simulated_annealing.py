from src.utils import get_distance

import numpy as np
from typing import List, Tuple


class SimulatedAnnealing:
    def __init__(
        self,
        initial_solution: List[Tuple[str, int, float, float]],
        temperature: int,
        anneal_rate: float,
        cooldown: float,
    ) -> None:
        """
        Initialize the simulated annealing algorithm

        :param initial_solution: initial solution
        :param temperature: initial temperature
        :param anneal_rate: annealing rate
        :param cooldown: final temperature
        """

        self.current_solution = initial_solution
        self.temperature = temperature
        self.anneal_rate = anneal_rate
        self.cooldown = cooldown
        self.best_solution = initial_solution
        self.best_cost = self.cost(initial_solution)
        self.cost_history = [self.best_cost]
        self.best_solution_history = [self.best_solution + self.best_solution[:1]]
        self.current_solution_history = [self.current_solution]
        self.iteration = 0

    def cost(self, solution: List[Tuple[str, int, float, float]]) -> float:
        """
        Calculate the total distance of a given solution

        :param solution: solution to calculate the cost
        :return: total distance of the solution
        """
        cost = 0
        for i in range(len(solution) - 1):
            cost += get_distance(
                solution[i][2], solution[i][3], solution[i + 1][2], solution[i + 1][3]
            )
        cost += get_distance(
            solution[-1][2], solution[-1][3], solution[0][2], solution[0][3]
        )
        return cost

    def acceptance_probability(
        self, candidate_cost: List[Tuple[str, int, float, float]]
    ) -> float:
        """
        Calculate the acceptance probability of a candidate solution

        :param candidate_cost: cost of the candidate solution
        :return: acceptance probability
        """
        return np.exp((self.best_cost - candidate_cost) / self.temperature)

    def accept(self, candidate: List[Tuple[str, int, float, float]]) -> None:
        """
        Accept the candidate solution if it is better than the current solution

        :param candidate: candidate solution
        """
        candidate_cost = self.cost(candidate)
        if candidate_cost < self.best_cost:
            self.best_cost = candidate_cost
            self.best_solution = candidate
        if candidate_cost < self.cost(self.current_solution):
            self.current_solution = candidate
        else:
            if np.random.random() < self.acceptance_probability(candidate_cost):
                self.current_solution = candidate

    def run(self) -> Tuple[List[Tuple[str, int, float, float]], float]:
        """
        Run the simulated annealing algorithm
        """
        while self.temperature > self.cooldown:
            candidate = self.current_solution.copy()
            i, j = np.random.randint(0, len(candidate), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            self.accept(candidate)
            self.temperature *= self.anneal_rate
            self.cost_history.append(self.best_cost)
            self.best_solution_history.append(
                self.best_solution + self.best_solution[:1]
            )
            self.current_solution_history.append(self.current_solution)
            self.iteration += 1
        return self.best_solution + self.best_solution[:1], self.best_cost

    def get_history(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Get the history of the algorithm

        :return: cost history, best solution history, current solution history
        """
        return (
            self.cost_history,
            self.best_solution_history,
            self.current_solution_history,
        )

    def get_iteration(self) -> int:
        """
        Get the number of iterations

        :return: number of iterations
        """
        return self.iteration

    def get_best_solution(self) -> List[Tuple[str, int, float, float]]:
        """
        Get the best solution

        :return: best solution
        """
        return self.best_solution + self.best_solution[:1]

    def get_best_cost(self) -> float:
        """
        Get the best cost

        :return: best cost
        """
        return self.best_cost
