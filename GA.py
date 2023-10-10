from typing import List, Callable, Tuple
from numpy.random import randint, rand
import numpy as np

# Objective function type definition
ObjectiveFunc = Callable[[List[float]], float]

# Objective function
def objective(x: List[float]) -> float:
    return x[0]**2.0 + x[1]**2.0

# Decode bitstring to numbers
def decode(bounds: List[Tuple[float, float]], n_bits: int, bitstring: List[int]) -> List[float]:
    decoded = []
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
    return decoded

# Tournament selection
def selection(pop: List[List[int]], scores: List[float], k: int = 3) -> List[int]:
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# Crossover two parents to create two children
def crossover(p1: List[int], p2: List[int], r_cross: float) -> List[List[int]]:
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1) - 2)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# Mutation operator
def mutation(bitstring: List[int], r_mut: float) -> None:
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]

# Genetic algorithm
def genetic_algorithm(objective_func: ObjectiveFunc, bounds: List[Tuple[float, float]], n_bits: int, n_iter: int, n_pop: int, r_cross: float, r_mut: float) -> Tuple[List[int], float]:
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = [], float('inf')

    for gen in range(n_iter):
        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = [objective_func(d) for d in decoded]

        # Elitism: Preserve the best individual
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f">{gen}, new best f({decoded[i]}) = {scores[i]}")

        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = []

        # Always include the best individual (elitism)
        children.append(best)

        while len(children) < n_pop:
            p1, p2 = selected[randint(len(selected))], selected[randint(len(selected))]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
                if len(children) >= n_pop:
                    break

        pop = children

    return best, best_eval

if __name__ == "__main__":
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]
    n_iter = 100
    n_bits = 16
    n_pop = 100
    r_cross = 0.9
    r_mut = 1.0 / (float(n_bits) * len(bounds))

    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    decoded = decode(bounds, n_bits, best)
    print(f'f({decoded}) = {score}')
