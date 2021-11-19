import numpy as np
import matplotlib.pyplot as plt


class KnapsackProblem:
    def __init__(self, amount_of_articles, max_weight, weight_range=(1, 11), value_range=(1, 11), rng_seed=None):
        """Generating objects, weights and values for each of them"""
        self.random_number_generator = np.random.default_rng(rng_seed)
        self.amount_of_articles = amount_of_articles
        self.max_weight = max_weight
        self.weights_of_articles = self.random_number_generator.integers(*weight_range, amount_of_articles)
        self.values_of_articles = self.random_number_generator.integers(*value_range, amount_of_articles)

    # generating random individual with 1 and 0 code for all articles
    def get_random_individual(self):
        return self.random_number_generator.integers(0, 2, self.amount_of_articles)

    def get_random_population(self, amount_of_individuals):
        return [self.get_random_individual() for _ in range(amount_of_individuals)]

    # Score is sum of values chosen by individual. If total weight is higher than weight limit then score is 0.
    def score_the_population(self, population):
        return [np.sum(np.multiply(population[i], self.values_of_articles)) if np.sum(
            np.multiply(population[i], self.weights_of_articles)) <= self.max_weight else 0 for i in
                range(len(population))]

    # Choosing top 50% best individuals using tournament method
    def get_parents_from_population_tournament_method(self, population):
        scores = self.score_the_population(population)
        order_of_competing = np.arange(len(population))
        self.random_number_generator.shuffle(order_of_competing)
        return [
            population[order_of_competing[i]] if scores[order_of_competing[i]] >= scores[order_of_competing[-i]] else
            population[order_of_competing[-i]] for i in range(len(population) // 2)]

    # TODO add other choosing methods

    # Crossover at rate 50% and 25%
    def get_children_from_two_parents(self, mom, dad, only_two=False):
        son1 = []
        son2 = []
        daughter1 = []
        daughter2 = []
        for i in range(len(mom)):
            rnd_number = self.random_number_generator.random()
            # Crossover rate 50%
            if rnd_number > 0.5:
                son1.append(mom[i])
                daughter1.append(dad[i])
            else:
                son1.append(dad[i])
                daughter1.append(mom[i])
            # Crossover rate 25%
            if rnd_number > 0.75:
                son2.append(mom[i])
                daughter2.append(dad[i])
            else:
                son2.append(dad[i])
                daughter2.append(mom[i])
        if only_two:
            return [son1, daughter1]
        return [son1, son2, daughter1, daughter2]

    def genotype_mutation_of_population(self, population, mutate_probability=0.001):
        genes = len(population) * len(population[0])
        for i in range(int(genes * mutate_probability)):
            random_selector = self.random_number_generator.integers(len(population))
            random_gene = self.random_number_generator.integers(len(population[random_selector]))
            if population[random_selector][random_gene] == 0:
                population[random_selector][random_gene] = 1
            else:
                population[random_selector][random_gene] = 0

        return population

    # Makes whole population of children from whole population of parents
    def get_all_children_from_all_parents(self, parents):
        order_of_crossing = np.arange(len(parents))
        self.random_number_generator.shuffle(order_of_crossing)
        children = []
        for i in range(0, len(parents), 2):

            # if not enough parents someone will be a polygamist
            if i + 1 == len(parents):
                children.extend(self.get_children_from_two_parents(parents[order_of_crossing[i]],
                                                                   self.random_number_generator.choice(parents), True))
            else:
                children.extend(self.get_children_from_two_parents(parents[order_of_crossing[i]],
                                                                   parents[order_of_crossing[i + 1]]))
        return children

    # Solves the problem
    def solve_problem(self, size_of_population, number_of_epochs, show_learning_curve=False):
        population = self.get_random_population(size_of_population)
        if show_learning_curve:
            best_scores = []
            worst_scores = []
            mean_scores = []

        for i in range(number_of_epochs):
            current_scores = self.score_the_population(population)

            if show_learning_curve:
                best_scores.append(np.max(current_scores))
                worst_scores.append(np.min(current_scores))
                mean_scores.append(np.mean(current_scores))

            print('Epoch {}: '.format(i), np.mean(current_scores))
            parents = self.get_parents_from_population_tournament_method(population)
            population = self.get_all_children_from_all_parents(parents)
            population = self.genotype_mutation_of_population(population, mutate_probability=0.001)

        if show_learning_curve:
            current_scores = self.score_the_population(population)
            best_scores.append(np.max(current_scores))
            worst_scores.append(np.min(current_scores))
            mean_scores.append(np.mean(current_scores))

            plt.figure("Learning curve")
            plt.plot(np.arange(len(best_scores)), best_scores, c='g')
            plt.plot(np.arange(len(best_scores)), worst_scores, c='r')
            plt.plot(np.arange(len(best_scores)), mean_scores, c='b')
            plt.legend(["Best scoring individual", "Worst scoring individual", "Mean score of population"])

        best_genotype = population[np.argmax(self.score_the_population(population))]
        print('Best genotype: ', best_genotype)
        return best_genotype


if __name__ == '__main__':
    ks = KnapsackProblem(amount_of_articles=100, max_weight=250)

    ks.solve_problem(size_of_population=150, number_of_epochs=200, show_learning_curve=True)
    plt.show()
