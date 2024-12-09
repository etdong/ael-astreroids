# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time
import EasyGA
import numpy as np
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from final_submission_controller import AEL_CompetitionController

def gene_generation():
    """
    Initializes a random set of membership functions for thrust
      - A Z function for poor
      - A triangular function for average
      - An S function for good

      Each function covers the input universe so not to require 27 rules for each.
    """

    # Generate random means for gaussian membership functions
    means = sorted(np.random.uniform(0.0, 1.0, 3).tolist())
    std_devs = np.random.uniform(0.01, 0.5, 3).tolist()

    return means, std_devs


def run_game(chromosome, graphics=False):
    # Define game scenario
    my_test_scenario = Scenario(name='Test Scenario',
                                num_asteroids=20,
                                ship_states=[
                                    {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                ],
                                map_size=(1000, 800),
                                time_limit=60,
                                ammo_limit_multiplier=0,
                                stop_if_no_ammo=False)

    # Define Game Settings
    game_settings = {'perf_tracker': True,
                     'graphics_type': GraphicsType.Tkinter,
                     'realtime_multiplier': 1,
                     'graphics_obj': None,
                     'frequency': 30}

    if graphics:
        game = KesslerGame(settings=game_settings)
    else:
        game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

    # Evaluate the game
    pre = time.perf_counter()

    run_score, perf_data = game.run(scenario=my_test_scenario, controllers=[AEL_CompetitionController(chromosome)])

    print("Time taken: ", time.perf_counter() - pre)
    return run_score

def fitness_function(chromosome):

    the_score = run_game(chromosome)

    return the_score.teams[0].deaths

ga = EasyGA.GA()
best_chromosome = None

choice = input("Do you want to run the GA? (y/n): \n"
               "Selecting No will run the best chromosome we found with the GA. It takes ~2 hours to run. \n")
if choice.lower() == 'y':

    ga.gene_impl = lambda: gene_generation()
    ga.chromosome_length = 2
    ga.population_size = 3
    ga.target_fitness_type = 'min'
    ga.max_generations = 2
    ga.fitness_function_impl = fitness_function

    ga.evolve()  # This takes like 2 hours to run
    best_chromosome = ga.sort_by_best_fitness()[0]

elif choice.lower() == 'n':
    gene1 = ([0.13770366912403575, 0.6214125400758881, 0.6463389796980702], [0.3825461461333196, 0.23969121472649166, 0.08229587488526449])
    gene2 = ([0.012369223596010115, 0.5344299502383826, 0.901989772978868], [0.49701069124218816, 0.14278966255682327, 0.4946189358260284])
    best_chromosome = ga.make_chromosome([gene1, gene2])

else:
    print("Invalid input. Exiting.")
    exit()


# Run the best
score = run_game(best_chromosome, True)

# Print out some general info about the result
print(score.stop_reason)
print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
print('Deaths: ' + str([team.deaths for team in score.teams]))
print('Accuracy: ' + str([team.accuracy for team in score.teams]))
print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
