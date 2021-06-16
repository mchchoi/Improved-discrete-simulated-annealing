# Based on https://github.com/jedrazb/python-tsp-simulated-annealing
import math
import random
import visualize_tsp
import matplotlib.pyplot as plt
import animated_visualizer


class SimAnneal(object):
    def __init__(self, coords, c,T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.c = c
        self.N = len(coords)
        self.T = math.sqrt(self.N)/math.log(2) if T == -1 else T
        #self.T = math.sqrt(self.N) /2 if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-16 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []
        self.solution_history =[]

    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        random.seed(2000)
        cur_node = random.choice(self.nodes)  # start from a random node
        solution = [cur_node]

        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def dist(self, node_0, node_1):
        """
        Euclidean distance between two nodes.
        """
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)

    def fitness(self, solution):
        """
        Total distance of the current solution path.
        """
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def improved_p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        #print(candidate_fitness)
        #print(self.c)
        #print(self.cur_fitness)
        self.c = candidate_fitness-5
        #self.c = candidate_fitness - 5
        #self.c = self.cur_fitness + 1
        if self.c >= candidate_fitness:
            return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)
        elif candidate_fitness > self.c and self.c >= self.cur_fitness:
            return (math.exp(-abs(self.c - self.cur_fitness) / self.T)) * self.T/(candidate_fitness-self.c+self.T)
        elif self.cur_fitness > self.c:
            return (self.cur_fitness-self.c+self.T)/(candidate_fitness-self.c+self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            #print(candidate_fitness)
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            #print(self.p_accept(candidate_fitness))
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def improved_accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
                self.c = self.best_fitness
        else:
            #print(self.improved_p_accept(candidate_fitness))
            if random.random() < self.improved_p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        #print(self.coords)
        self.cur_solution, self.cur_fitness = self.initial_solution()
        self.solution_history.append(self.cur_solution)

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            self.accept(candidate)
            #self.T *= self.alpha
            self.iteration += 1
            self.T = math.sqrt(self.N)/math.log(self.iteration+1)
            #self.T = math.sqrt(self.N) / (self.iteration + 1)
            self.fitness_list.append(self.cur_fitness)
            self.solution_history.append(self.cur_solution)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def improved_anneal(self):
        """
        Execute improved simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()
        self.c = self.cur_fitness
        self.solution_history.append(self.cur_solution)

        print("Starting improved annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            self.improved_accept(candidate)
            #self.T *= self.alpha
            self.iteration += 1
            self.T = math.sqrt(self.N)/math.log(self.iteration+1)
            self.solution_history.append(self.cur_solution)
            #self.T = math.sqrt(self.N)/ (self.iteration + 1)
            #self.T = 1*math.exp(-(self.iteration + 1))

            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print(f"Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    def visualize_routes(self):
        """
        Visualize the TSP route with matplotlib.
        """
        visualize_tsp.plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()

    def animateSolutions(self,coords):
        animated_visualizer.animateTSP(self.solution_history, coords)
