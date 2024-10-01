import random
from deap import base, creator, tools
# `model_train` contains functions related to model training
from model_train import *
import model_train

# Define the search space for hyperparameters
param_space = {
    'EPOCHS': (5, 200),
    'NUM_SAMPLES': (1, 10000),
    'LEARNING_RATE': (1e-5, 1e-2),
    'MAX_LR': (1e-4, 1e-1),
    'TRAIN_BATCH_SIZE': (10, 2000),
    'NOISE_MULTIPLIER': (0.5, 10.0),
    'NUM_EXPERTS': (1, 5),
    'NEURON_NUM_0': (5, 1000),
    'NEURON_NUM_1': (5, 1000),
    'NEURON_NUM_2': (5, 1000),
    'DROPOUT_PROB': (0.0, 0.5),
    'LOSS_FUNCTION': (0, 2),  # 0: MSE, 1: MAE, 2: HuberLoss
}

# Use DEAP to create individuals and populations
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Initialize the hyperparameters
for param, (low, high) in param_space.items():
    if isinstance(low, int) and isinstance(high, int):
        toolbox.register(f"attr_{param}", random.randint, low, high)
    else:
        toolbox.register(f"attr_{param}", random.uniform, low, high)

# Define the individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 [toolbox.attr_EPOCHS, toolbox.attr_NUM_SAMPLES, toolbox.attr_LEARNING_RATE,
                  toolbox.attr_MAX_LR, toolbox.attr_TRAIN_BATCH_SIZE, toolbox.attr_NOISE_MULTIPLIER,
                  toolbox.attr_NUM_EXPERTS, toolbox.attr_NEURON_NUM_0, toolbox.attr_NEURON_NUM_1,
                  toolbox.attr_NEURON_NUM_2, toolbox.attr_DROPOUT_PROB, toolbox.attr_LOSS_FUNCTION], n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function
def evaluate(individual):
    # Read parameters from individual
    (model_train.EPOCHS, model_train.NUM_SAMPLES, model_train.LEARNING_RATE, model_train.MAX_LR, model_train.TRAIN_BATCH_SIZE,
     model_train.NOISE_MULTIPLIER, model_train.NUM_EXPERTS, NEURON_NUM_0, NEURON_NUM_1, NEURON_NUM_2,
     model_train.DROPOUT_PROB, model_train.LOSS_FUNCTION, model_train.CASE_INDEX) = individual

    # Define the structure of each part in the whole model
    model_train.NET_CONFIG = {'0': NEURON_NUM_0, '1': NEURON_NUM_1, '2': NEURON_NUM_2}
    model_train.TEST_BATCH_SIZE = model_train.TRAIN_BATCH_SIZE
    model_train.IS_LABELS_NORMALIZED = True
    # Define the dataset CASE options
    model_train.CASE = 'uniform3'
    # Enable DP Index
    model_train.PRIVACY = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get train data loader and test data loader
    test_loader, train_loader = prepare_loaders()

    # Create model and train it
    model = create_model(device)
    train_model(model, train_loader, device, test_loader)

    # Evaluate the model's performance
    avg_error, max_error, max_error_example, correct_example, avg_val_loss = evaluate_model(model, test_loader, device)

    return avg_val_loss,  # Return the evaluation result as the fitness


# Crossover, mutation, and selection operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[10, 1, 1e-5, 1e-4, 10, 0.5, 1, 0.0, 0, 0],
                 up=[100, 10000, 1e-3, 1e-1, 5000, 10.0, 5, 0.5, 2, 4], eta=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


# Run the evolutionary search
def evolutionary_search():
    population = toolbox.population(n=10)  # Population size
    ngen = 20  # Number of generations
    cxpb = 0.5  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Evaluate the initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Start evolution
    for gen in range(ngen):
        print(f"Generation {gen}")

        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population
        population[:] = offspring

    # Select the best individual
    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_ind}, Fitness: {best_ind.fitness.values}")


if __name__ == "__main__":
    evolutionary_search()