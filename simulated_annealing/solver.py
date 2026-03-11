from random import uniform
from .temperature_functions import *
from .probability_functions import *

def generic_solver_factory(
    get_initial_state,
    get_random_neighbour,
    state_to_energy,
    calculate_temperature=temperature_standard,
    acceptance_prob_function=probability_standard,
    debug_mode=False,
    return_history=False
):
    def simulated_annealing(k_max):
        s = get_initial_state()
        energy_of_s = state_to_energy(s)

        best_s = s
        best_energy = energy_of_s

        history = {
            'temperature': [],
            'current_energy': [],
            'best_energy': [],
            'state': []
        }

        if debug_mode:
            print('The maximum k is:', k_max)
            print('The initial state is:', s)
            print('------------------------------------------------------')

        for k in range(k_max):
            temp = calculate_temperature(1 - ((k + 1) / k_max))

            s_new = get_random_neighbour(s)
            energy_of_s_new = state_to_energy(s_new)

            res_acceptance_prob_function = acceptance_prob_function(
                energy_of_s, energy_of_s_new, temp
            )
            limit = uniform(0, 1)

            if res_acceptance_prob_function >= limit:
                s = s_new
                energy_of_s = energy_of_s_new

                if energy_of_s < best_energy:
                    best_s = s
                    best_energy = energy_of_s

            if return_history:
                history['temperature'].append(temp)
                history['current_energy'].append(energy_of_s)
                history['best_energy'].append(best_energy)
                history['state'].append(s)

        if return_history:
            return best_s, history

        return best_s

    return simulated_annealing