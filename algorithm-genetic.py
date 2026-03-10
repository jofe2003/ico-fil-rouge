from abc import ABC, abstractmethod
class GeneticAlgorithm(ABC):
    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def evaluate_fitness(self, individual):
        pass

    @abstractmethod
    def select_parents(self):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, individual):
        pass

    @abstractmethod
    def run(self):
        pass