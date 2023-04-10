# -*- coding: utf-8 -*-
"""Regulate class that creates simulation and grid data instances."""

import logging
import math
import time
import random

from db_connector import DBConnector
from data_extractor import DataExtractor
from logger import Logger
from simulation import GTNETSimulation, RuntimeSimulation


# pylint: disable=line-too-long
def calculate_rms_current(power, voltage):
    """Calculate the RMS currents. I = S/(sqrt(3)*U).

    :param list power: The RMS powers.
    :param list voltage: The RMS voltages.
    :return The RMS currents.
    :rtype list[float]
    """
    rms_power_voltage = []
    for rms_power, rms_voltage in zip(power, voltage):
        rms_power_voltage.append(rms_power / (math.sqrt(3) * rms_voltage))
    return rms_power_voltage


class RuntimeRegulation:
    """Regulate class."""

    def __init__(self, ip, port, table_name=None):
        """initialize class.
        :param str ip: IP address.
        :param int port: Port number.
        :param str table_name: Name of the table to store or read the results from.
        """
        self.logger = logging.getLogger("Regulation")
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger())
        self.logger.addHandler(stream_handler)

        self.simulation = RuntimeSimulation(ip, port)

        self.data = DataExtractor()
        if table_name is not None:
            self.table_name = table_name
            self.database = DBConnector("Database.db")
            self.load_index, self.database_dict = self.data.get_database_dict()
            self.database.add_table(self.table_name, self.database_dict)

    def one_gen_regulation(self, generator, regulation_value, reference_key, reference_value, step=0.001, offset=.1,
                           delay=0):
        """Regulate one generator.

        :param str generator: Generator name.
        :param regulation_value:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """
        self.logger.debug(f"Regulating generator: {generator}")
        path = self.data.grid_data["generators"][generator][regulation_value]["path"]
        initial_value = self.data.grid_data["generators"][generator][regulation_value]["InitValue"]
        min_value = self.data.grid_data["generators"][generator][regulation_value]["Min"]
        max_value = self.data.grid_data["generators"][generator][regulation_value]["Max"]
        results = []
        first_res = self.simulation.read_value(reference_key)
        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return True
        results.append(first_res)
        current_value = initial_value
        direction = True
        i = 0
        while min_value <= current_value <= max_value:
            current_res = self.simulation.set_value_and_read_output({path: current_value}, [reference_key], delay)
            i += 1
            if not current_res:
                self.logger.debug("Exiting the program...")
                exit()

            if delay == -1:
                current_res = self.__dynamic_stabilisation(reference_key, current_res)
            results.append(current_res)

            if reference_value - offset < current_res < reference_value + offset:
                self.logger.info(f"Optimal value found!")
                self.logger.debug(
                    f"The following value has been adjusted: \n \t\t\t\t {generator}: {regulation_value} from {initial_value} to {current_value} \n \t\t\t\t{reference_key} = {current_res}")

                return current_value
            if direction:
                if current_res < reference_value:
                    step = step
                elif current_res > reference_value:
                    current_value = initial_value
                    step = -abs(step)
                direction = False

            if (results[-2] < reference_value < results[-1]) or (
                    results[-2] > reference_value > results[-1]
            ):
                self.logger.debug(
                    f"The following value has been adjusted: \n \t\t\t\t {generator}: {regulation_value} from {initial_value} to {current_value} \n \t\t\t\t {reference_key} = {current_res}")
                return current_value
            if current_value + step > max_value or current_value + step < min_value:
                self.logger.warning(f"{regulation_value} max or min value reached!")
                self.logger.debug(
                    f"The following value has been adjusted: \n \t\t\t\t {generator}: {regulation_value} from {initial_value} to {current_value} \n \t\t\t\t {reference_key} = {current_res}")
                return current_value
            else:
                current_value += step

    def dynamic_regulation(self, generators=None, regulation_values=None, reference_key="W3", reference_value=377,
                           step=0.001, offset=.1,
                           delay=0):
        """Regulate one generator.

        :param list generators: Generator name.
        :param list regulation_values:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """
        self.logger.info("Start dynamic regulation")
        if not generators:
            generators = list(self.data.grid_data["generators"].keys())
        if not regulation_values:
            regulation_values = self.data.find_gen_pv_data(generators)
        gen_list = []
        current_value_list = []
        regulate = []
        for (generator, regulation_value) in zip(generators, regulation_values):
            gens = []
            path = self.data.grid_data["generators"][generator][regulation_value]["path"]
            initial_value = self.data.grid_data["generators"][generator][regulation_value]["InitValue"]
            min_value = self.data.grid_data["generators"][generator][regulation_value]["Min"]
            max_value = self.data.grid_data["generators"][generator][regulation_value]["Max"]
            gens.append(path)
            gens.append(initial_value)
            gens.append(min_value)
            gens.append(max_value)
            regulate.append(True)
            current_value_list.append(initial_value)
            gen_list.append(gens)

        if isinstance(step, list):
            for index, value in enumerate(step):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(step)
        if isinstance(delay, list):
            for index, value in enumerate(delay):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(delay)
        results = []
        first_res = self.simulation.read_value(reference_key)
        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return current_value_list
        results.append(first_res)
        direction = True
        while True in regulate:
            for index, gen in enumerate(gen_list):
                step = gen[4]
                delay = gen[5]
                min_value = gen[2]
                max_value = gen[3]
                if min_value <= current_value_list[index] <= max_value:
                    current_res = self.simulation.set_value_and_read_output({gen[0]: current_value_list[index]},
                                                                            [reference_key], delay)
                    if delay == -1:
                        self.__dynamic_stabilisation(reference_key, current_res)
                    results.append(current_res)
                    if reference_value - offset < current_res < reference_value + offset:
                        self.logger.info("Optimal point found")
                        result = ""
                        for ind, generator in enumerate(gen_list):
                            result += f"\t\t\t\t {generators[ind]}:  {regulation_values[ind]} from {generator[1]} to {current_value_list[ind]} \n "

                        self.logger.info(
                            f"The following value has been adjusted: \n {result} \n \t\t\t\t{reference_key} = {current_res}")
                        return current_value_list

                    if direction:
                        if current_res < reference_value:
                            step = step
                        elif current_res > reference_value:
                            current_value_list[index] = gen[1]
                            step = -abs(step)
                        direction = False
                    if (results[-2] < reference_value < results[-1]) or (
                            results[-2] > reference_value > results[-1]
                    ):
                        result = ""
                        for ind, generator in enumerate(gen_list):
                            result += f"\t\t\t\t {generators[ind]}:  {regulation_values[ind]} from {generator[1]} to {current_value_list[ind]} \n "

                        self.logger.info(
                            f"The following value has been adjusted: \n {result} \n \t\t\t\t{reference_key} = {current_res}")
                        return current_value_list
                    if current_value_list[index] + step > max_value or current_value_list[index] + step < min_value:
                        result = ""
                        for ind, generator in enumerate(gen_list):
                            self.logger.warning(f"{regulation_values[ind]} max or min value reached!")
                            result += f"\t\t\t\t {generators[ind]}:  {regulation_values[ind]} from {generator[1]} to {current_value_list[ind]} \n "

                        self.logger.info(
                            f"The following value has been adjusted: \n {result} \n \t\t\t\t{reference_key} = {current_res}")
                        return current_value_list
                    else:
                        current_value_list[index] += step
                    # current_value_list[index] += step
                else:
                    regulate[index] = False
        return print("not found!")

    def power_constraint_regulation(self, generators=None, regulation_keys=None, reference_key="W1",
                                    reference_value=377,
                                    rated_power=None,
                                    active_and_reactive_names=None,
                                    step=0.001, offset=.1, delay=0):
        """Regulate multiple generators specifically.

        :param active_and_reactive_names:
        :param rated_power:
        :param list generators: Generator name.
        :param list regulation_keys:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """
        self.logger.info("Start dynamic power constrained regulation")
        if not generators:
            generators = list(self.data.grid_data["generators"].keys())
        if not regulation_keys:
            regulation_keys = self.data.find_gen_pv_data(generators)
        if not rated_power:
            rated_power = self.data.get_rms_power(generators)
        if not active_and_reactive_names:
            active_and_reactive_names = self.data.get_generator_powers_name(generators)
        for (gen, max_p, p_v_names) in zip(generators, rated_power, active_and_reactive_names):
            self.logger.debug(
                f"{gen} has a rated power of {max_p}, with active power name '{p_v_names[0]}' and reactive power '{p_v_names[1]}'.")
        gen_list = []
        current_value_list = []
        regulate = []
        for (generator, regulation_value) in zip(generators, regulation_keys):
            gens = []
            path = self.data.grid_data["generators"][generator][regulation_value]["path"]
            initial_value = self.simulation.read_value(regulation_value, "SliderCapture")
            min_value = self.data.grid_data["generators"][generator][regulation_value]["Min"]
            max_value = self.data.grid_data["generators"][generator][regulation_value]["Max"]
            gens.append(path)
            gens.append(initial_value)
            gens.append(min_value)
            gens.append(max_value)
            regulate.append(True)
            current_value_list.append(initial_value)
            gen_list.append(gens)
        if isinstance(step, list):
            for index, value in enumerate(step):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(step)
        if isinstance(delay, list):
            for index, value in enumerate(delay):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(delay)
        results = []

        first_res = self.simulation.read_value(reference_key)
        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return current_value_list

        results.append(first_res)

        direction = True
        if reference_value < first_res:
            direction = False
        while True in regulate:
            old_diff = 0
            for i, rated_power_i in enumerate(rated_power):
                active_name, reactive_name = active_and_reactive_names[i]
                active_value = self.simulation.read_value(active_name)
                reactive_value = self.simulation.read_value(reactive_name)
                power = abs(complex(active_value, reactive_value))
                current_diff = power - rated_power_i
                if regulate[i]:
                    if direction and current_diff < old_diff:
                        old_diff = current_diff
                        index = i
                    elif not direction and current_diff > old_diff:
                        old_diff = current_diff
                        gen_list[i][4] = -step
                        index = i
            gen = gen_list[index]
            _step = gen[4]
            delay = gen[5]
            if regulate[index]:
                current_value_list[index] = round(current_value_list[index] + _step, 6)
                current_res = self.simulation.set_value_and_read_output({gen[0]: current_value_list[index]},
                                                                        [reference_key], delay)
                results.append(current_res)
                if reference_value - offset < current_res < reference_value + offset:
                    self.logger.info("Optimal point found")
                    result = ""
                    for ind, gen in enumerate(gen_list):
                        result += f"{generators[ind]}:  {regulation_keys[ind]} changed from {gen[1]} to {current_value_list[ind]} \n "
                    self.logger.info(f"The following value has been adjusted: \n {result}")
                    return current_value_list

                if (results[-2] < reference_value < results[-1]) or (
                        results[-2] > reference_value > results[-1]
                ):
                    return current_value_list
                if gen[3] <= current_value_list[index] + _step or current_value_list[index] + _step <= gen[2]:
                    regulate[index] = False
                    return

        return logging.warning("Optimal value not found!")

    def __current_constraint_regulation(self, generators, reference_key, reference_value, offset, regulate,
                                        rms_current_names, max_current, gen_list, step, current_value_list,
                                        regulation_values):
        """Regulate multiple generators specifically.

        :param u:
        :param rms_current_names:
        :param optimal_power:
        :param list generators: Generator name.
        :param list regulation_values:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """

        results = []
        first_res = self.simulation.read_value(reference_key)

        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return current_value_list
        results.append(first_res)
        k = 0

        direction = True
        if reference_value < first_res:
            direction = False
        while True in regulate:
            old_diff = 0
            for i, current_i in enumerate(rms_current_names):
                current_rms = self.simulation.read_value(current_i)
                current_diff = current_rms - max_current[i]
                if direction and regulate[i]:
                    if current_diff < old_diff:
                        old_diff = current_diff
                        index = i
                elif not direction and regulate[i]:
                    if current_diff > old_diff:
                        old_diff = current_diff
                        gen_list[i][4] = -step
                        index = i
            gen = gen_list[index]
            _step = gen[4]
            delay = gen[5]
            if regulate[index]:
                current_value_list[index] = round(current_value_list[index] + _step, 6)
                current_res = self.simulation.set_value_and_read_output({gen[0]: current_value_list[index]},
                                                                        [reference_key], delay)
                results.append(current_res)
                if reference_value - offset < current_res < reference_value + offset:
                    self.logger.info("Optimal point found")
                    result = ""
                    for ind, gen in enumerate(gen_list):
                        result += f"{generators[ind]}:  {regulation_values[ind]} changed from {gen[1]} to {current_value_list[ind]} \n "
                    self.logger.info(f"The following value has been adjusted: \n {result}")
                    return current_value_list

                if (results[-2] < reference_value < results[-1]) or (
                        results[-2] > reference_value > results[-1]
                ):
                    return current_value_list
                if gen[3] <= current_value_list[index] + _step or current_value_list[index] + _step <= gen[2]:
                    regulate[index] = False
                    return current_value_list
        logging.warning("Optimal value not found!")
        return current_value_list

    def current_constraint_regulation(self, generators=None, regulation_keys=None, reference_key="W1",
                                      reference_value=377,
                                      rated_power=None,
                                      rms_current_names=None, voltages=None, step=0.001, offset=.1, delay=0,
                                      keep_reg=False):
        """Regulate generators based on their RMS current.

        :param list generators: List of the generators to be regulated.
        :param list regulation_keys: The parameters to be adapted.
        :param str reference_key: The reference parameter to be checked.
        :param float reference_value: The value of the reference parameter to be reached.
        :param list rated_power: List of the rated power.
        :param list rms_current_names: List of the RMS slider names in the runtime.
        :param list voltages: List of the RMS Line-to-Line voltages.
        :param float_or_list step: Increase or decrease step.
        :param float_or_list offset: Acceptable offset to the reference value.
        :param float_or_list delay: Delay between setting and reading the value.
        :param bool keep_reg: If True, keep adjusting the regulation_keys to adapt the RMS currents.
        :return:
        """
        self.logger.info("Start dynamic regulation")
        if not generators:
            generators = list(self.data.grid_data["generators"].keys())
        if not regulation_keys:
            regulation_keys = self.data.find_gen_pv_data(generators)
        if not rms_current_names:
            rms_current_names = self.data.get_rms_currents(generators)
        if not rated_power:
            rated_power = self.data.get_rms_power(generators)
        if not voltages:
            voltages = self.data.get_rated_voltage(generators)

        max_current = calculate_rms_current(rated_power, voltages)
        for (gen, max_i) in zip(generators, max_current):
            self.logger.debug(
                f"{gen} has an RMS current of {max_i} kA.")
        gen_list = []
        current_value_list = []
        regulate = []
        for (generator, regulation_value) in zip(generators, regulation_keys):
            gens = []
            path = self.data.grid_data["generators"][generator][regulation_value]["path"]
            initial_value = self.simulation.read_value(regulation_value, "SliderCapture")
            min_value = self.data.grid_data["generators"][generator][regulation_value]["Min"]
            max_value = self.data.grid_data["generators"][generator][regulation_value]["Max"]
            gens.append(path)
            gens.append(initial_value)
            gens.append(min_value)
            gens.append(max_value)
            regulate.append(True)
            current_value_list.append(initial_value)
            gen_list.append(gens)
        if isinstance(step, list):
            for index, value in enumerate(step):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(step)
        if isinstance(delay, list):
            for index, value in enumerate(delay):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(delay)
        current_value_list = self.__current_constraint_regulation(generators, reference_key, reference_value, offset,
                                                                  regulate,
                                                                  rms_current_names, max_current, gen_list, step,
                                                                  current_value_list,
                                                                  regulation_keys)
        if not keep_reg:
            return

        self.logger.debug("Try to adjust the RMS currents.")
        k = 0
        while True:
            current_diff = []
            curr_rms_res = []
            for i, current_i in enumerate(rms_current_names):
                current_rms = self.simulation.read_value(current_i)
                curr_rms_res.append(current_rms)
                current_diff.append(current_rms - max_current[i])

            if all(val > 0 for val in current_diff) or all(val < 0 for val in current_diff):
                self.logger.debug("Cannot adjust the values anymore.")
                result = ""
                for ind, gen in enumerate(gen_list):
                    result += f"{generators[ind]}:  {regulation_keys[ind]} changed from {gen[1]} to {current_value_list[ind]}, I = {curr_rms_res[ind]} kV \n "
                self.logger.info(f"Final values: \n {result}")
                return
            lowest_diff = 0
            highest_diff = 0
            for i, diff in enumerate(current_diff):
                if diff > highest_diff:
                    highest_diff = diff
                    current_value_list[i] -= step
                    highest_val = current_value_list[i]
                    high_gen = generators[i]
                    high_gen_regulation_value = regulation_keys[i]
                if diff < lowest_diff:
                    lowest_diff = diff
                    current_value_list[i] += step
                    lowest_val = current_value_list[i]

                    low_gen = generators[i]
                    low_gen_regulation_value = regulation_keys[i]

            print(high_gen, highest_diff, low_gen, lowest_diff, reference_key)
            # path = self.data.grid_data["generators"][high_gen][high_gen_regulation_value]["path"]
            # print(path), plot="RMS", plot_save_name=k
            print("Decrease", high_gen, highest_val, highest_diff, k)
            current_res = self.simulation.set_value_and_read_output({path: highest_val}, [reference_key])
            k += 1
            if reference_value - offset < current_res < reference_value + offset or current_res > reference_value + offset:
                self.logger.info("Here")
                continue

            self.simulation.set_value({path: lowest_val})

            k += 1

        return

    def predictive_regulation(self, diff=5, first=True, regulate=False):
        """Predict the best grid parameter based on previous runs and optionally continue regulating

        :param int diff: Maximum difference between current and stored data.
        :param boo first: Grab first match if True.
        :param bool regulate: Continue regulation if set to True.
        """
        res = self.find_matches(diff, first)
        if res:
            headers = self.database.headers(self.table_name)[self.load_index:]
            for index, gen in enumerate(self.data.grid_data["generators"].values()):
                self.simulation.set_value({gen[headers[index]]["path"]: res[index]})
        if not regulate:
            regulate = input('Continue regulation (Y/N): ')
            if regulate.lower() == 'y':
                regulate = True
            elif regulate.lower() == 'n':
                regulate = False
            else:
                print('Type Y or N')
        if regulate:
            reg_method = input(
                "which method [1:one_gen_regulation, 2:dynamic_regulation, 3:power_constraint_regulation,4:current_constraint_regulation]")
            if int(reg_method) == 1:
                print('one_gen_regulation')
                self.one_gen_regulation()
            elif int(reg_method) == 2:
                print('dynamic regulation')
                self.dynamic_regulation()
            elif int(reg_method) == 3:
                print('power_constraint_regulation')
                self.power_constraint_regulation()
            elif int(reg_method) == 4:
                print('current_constraint_regulation')
                self.current_constraint_regulation()
            else:
                print('Type 1 2 3 or 4')

    def find_matches(self, diff, first):
        """Find matches in the data base.

        :param int diff: Maximum difference between current and stored data.
        :param boo first: Grab first match if True.
        :return list of the matched data.
        :rtype list
        """
        headers = self.database.headers(self.table_name)[:self.load_index]
        database_data = self.database.read_data(self.table_name)
        old_difference = diff * (self.load_index + 1)
        data = 0
        for stored_data in database_data:
            if all(abs(stored_data[index] - self.simulation.read_value(headers[index], "SliderCapture")) > diff for
                   index in range(self.load_index)):
                continue
            current_diff = sum(
                [abs(stored_data[index] - self.simulation.read_value(headers[index], "SliderCapture")) > diff for
                 index in range(self.load_index)])
            if first:
                self.logger.info(
                    f"Found first matching values in the database, setting {self.database.headers(self.table_name)[self.load_index:]} to {stored_data[self.load_index:]}")
                return stored_data[self.load_index:]
            if current_diff < old_difference:
                old_difference = current_diff
                data = stored_data
        if old_difference == diff * (self.load_index + 1):
            self.logger.debug(f"No values found in {self.table_name}")
            return False
        self.logger.info(
            f"Found values in the database, setting {self.database.headers(self.table_name)[self.load_index:]} to {data[self.load_index:]}")
        return data[self.load_index:]

    def save_results(self):
        """Save results in the data base."""
        data = []
        for param in self.database_dict:
            data.append(self.simulation.read_value(param, "SliderCapture"))
        self.database.add_data_to_table(self.table_name, data)

    def __dynamic_stabilisation(self, parameter, old_value, sensitivity=0.09):
        """Wait for the system to reach a stable point.

        :param str parameter: Parameter to be stabilised.
        :param int old_value: Old parameter output.
        :param float sensitivity: Difference range between the old and the new value.
        """
        new_value = self.simulation.read_value(parameter)
        while abs(old_value - new_value) > sensitivity:
            old_value = new_value
            time.sleep(1)
            new_value = self.simulation.read_value(parameter)
        return new_value

    def inject_load_fault(self, load=None, parameter="Pset", new_value=None, constraints=None):
        """Inject load fault."""
        if load is None:
            self.logger.info("Random fault injection in progress...")
            load = random.choice(list(self.data.grid_data["loads"]))
            pset = self.data.load_active_reactive_power([load])[0][0]
            if constraints is None and new_value is None:
                constraints = [self.data.grid_data["loads"][load][pset]["Min"],
                               self.data.grid_data["loads"][load][pset]["Max"]]
                new_value = round(
                    random.uniform(constraints[0], constraints[1]), 2)
                self.simulation.set_value({self.data.grid_data["loads"][load][pset]["path"]: new_value})
                self.logger.info(
                    f"Setting the value of {load} from {self.data.grid_data['loads'][load][pset]['InitValue']} to {new_value}")
            else:
                self.simulation.set_value({self.data.grid_data["loads"][load][pset]["path"]: new_value})
                self.logger.info(
                    f"Setting the value of {load} from {self.data.grid_data['loads'][load][pset]['InitValue']} to {new_value}")

    def inject_short_circuit_fault(self, switches):
        """Inject short circuit fault."""
        self.simulation.set_switch_value(switches)


class GTNETRegulation:
    """Regulate class."""

    def __init__(self, ip, port, table_name=None, gen_pref_names=None):
        """initialize class.
        :param str ip: IP address.
        :param int port: Port number.
        :param str table_name: Name of the table to store or read the results from.
        :param list gen_pref_names: List of reference P of the generators.
        """
        self.logger = logging.getLogger("Regulation")
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger())
        self.logger.addHandler(stream_handler)

        self.simulation = GTNETSimulation(ip, port)

        self.data = DataExtractor()
        if table_name is not None:
            self.table_name = table_name
            self.database = DBConnector("Database.db")
            self.logger.info("Connected successful to the database Database.db")
            if self.database.check_table_exists(table_name):
                return
            if gen_pref_names is None:
                self.logger.error(f"Cannot create table {table_name}, please provide 'gen_pref_names' parameter")
                return
            self.load_index, self.database_dict = self.data.get_database_dict_GTNET(gen_pref_names)
            self.database.add_table(self.table_name, self.database_dict)

    def one_gen_regulation(self, generator, initial_value=0, min_max=None, reference_value=377, step=0.001, offset=.1,
                           delay=0):
        """Regulate one generator.

        :param initial_value:
        :param min_max:
        :param str generator: Generator name.
        :param regulation_value:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """
        if min_max is None:
            min_max = [0, 1]
        self.logger.debug(f"Regulating generator: {generator}")

        min_value = min_max[0]
        max_value = min_max[1]
        results = []
        first_res = self.simulation.set_value_and_read_output([initial_value], self.data.unpacked_types)[0]
        first_res = self.__dynamic_stabilisation([initial_value], first_res)
        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return True
        results.append(first_res)
        current_value = initial_value

        if reference_value < first_res:
            step = -step
        while min_value <= current_value <= max_value:
            current_value = round(current_value + step, 6)
            current_res = self.simulation.set_value_and_read_output([current_value], self.data.unpacked_types)[0]

            if not current_res:
                self.logger.debug("Exiting the program...")
                exit()

            if delay == -1:
                current_res = self.__dynamic_stabilisation([current_value], current_res)
            else:
                time.sleep(delay)
            results.append(current_res)
            if reference_value - offset < current_res < reference_value + offset:
                self.logger.info(f"Optimal value found!")
                self.logger.debug(
                    f"The following value has been adjusted: \n \t\t\t\t {generator}: {self.data.gtnet['input']} from {initial_value} to {current_value} \n \t\t\t\t{self.data.gtnet['output']} = {current_res}")

                return current_value

            if (results[-2] < reference_value < results[-1]) or (
                    results[-2] > reference_value > results[-1]
            ):
                self.logger.debug(
                    f"The following value has been adjusted: \n \t\t\t\t {generator}: {self.data.gtnet['input']} from {initial_value} to {current_value} \n \t\t\t\t {self.data.gtnet['output']} = {current_res}")
                return current_value
            if current_value + step > max_value or current_value + step < min_value:
                self.logger.warning(f"{self.data.gtnet['input']} max or min value reached!")
                self.logger.debug(
                    f"The following value has been adjusted: \n \t\t\t\t {generator}: {self.data.gtnet['input']} from {initial_value} to {current_value} \n \t\t\t\t {self.data.gtnet['output']} = {current_res}")
                return current_value

    def __dynamic_stabilisation(self, current_value_list, old_value, sensitivity=0.09):
        """Wait for the system to reach a stable point.

        :param list current_value_list: Parameter to be stabilised.
        :param int old_value: Old parameter output.
        :param float sensitivity: Difference range between the old and the new value.
        """
        time.sleep(1)
        new_value = self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[0]
        while abs(old_value - new_value) > sensitivity:
            old_value = new_value
            time.sleep(1)
            new_value = self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[0]
        return new_value

    def dynamic_regulation(self, generators, min_max, initial_values, result_index=0, reference_value=377,
                           step=0.001, offset=.1,
                           delay=0):
        """Regulate one generator.

        :param list generators: Generator name.
        :param list regulation_values:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """
        self.logger.info("Start dynamic regulation")
        gen_list = []
        current_value_list = []
        regulate = []
        for (generator, mn, initial_value) in zip(generators, min_max, initial_values):
            gens = []
            min_value = mn[0]
            max_value = mn[1]
            gens.append(generator)
            gens.append(initial_value)
            gens.append(min_value)
            gens.append(max_value)
            regulate.append(True)
            current_value_list.append(initial_value)
            gen_list.append(gens)

        if isinstance(step, list):
            for index, value in enumerate(step):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(step)
        if isinstance(delay, list):
            for index, value in enumerate(delay):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(delay)
        results = []
        first_res = self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[
            result_index]
        first_res = self.__dynamic_stabilisation(current_value_list, first_res)
        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return current_value_list
        results.append(first_res)
        direction = True

        while True in regulate:
            for index, gen in enumerate(gen_list):
                step = gen[4]
                delay = gen[5]
                min_value = gen[2]
                max_value = gen[3]
                if min_value <= current_value_list[index] <= max_value:
                    current_res = \
                        self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types, delay)[
                            result_index]
                    if delay == -1:
                        current_res = self.__dynamic_stabilisation(current_value_list, current_res)
                    results.append(current_res)
                    if reference_value - offset < current_res < reference_value + offset:
                        self.logger.info("Optimal point found")
                        result = ""
                        for ind, generator in enumerate(gen_list):
                            result += f"\t\t\t\t {generators[ind]}:  {self.data.gtnet['input'][ind]} from {generator[1]} to {current_value_list[ind]} \n "

                        self.logger.info(
                            f"The following value has been adjusted: \n {result} \n reference_key\t\t\t\t{1} = {current_res}")
                        return current_value_list

                    if direction:
                        if current_res < reference_value:
                            step = step
                        elif current_res > reference_value:
                            current_value_list[index] = gen[1]
                            step = -abs(step)
                        direction = False
                    if (results[-2] < reference_value < results[-1]) or (
                            results[-2] > reference_value > results[-1]
                    ):
                        result = ""
                        for ind, generator in enumerate(gen_list):
                            result += f"\t\t\t\t {generators[ind]}: {self.data.gtnet['input'][ind]} from {generator[1]} to {current_value_list[ind]} \n "

                        self.logger.info(
                            f"The following value has been adjusted: \n {result} \n \t\t\t\t{self.data.gtnet['output'][ind]} = {current_res}")
                        return current_value_list
                    if current_value_list[index] + step > max_value or current_value_list[index] + step < min_value:
                        result = ""
                        for ind, generator in enumerate(gen_list):
                            self.logger.warning(f"{1}regulation_values[ind] max or min value reached!")
                            result += f"\t\t\t\t {generators[ind]}: {self.data.gtnet['input'][ind]} from {generator[1]} to {current_value_list[ind]} \n "

                        self.logger.info(
                            f"The following value has been adjusted: \n {result} \n \t\t\t\t{self.data.gtnet['output'][ind]} = {current_res}")
                        return current_value_list
                    else:
                        current_value_list[index] += step
                else:
                    regulate[index] = False
        return print("not found!")

    def power_constraint_regulation(self, generators, min_max, initial_values, active_and_reactive_index,
                                    rated_power=None, result_index=0, reference_value=377, step=0.001, offset=.1,
                                    delay=0):
        """Regulate multiple generators specifically.

        :param active_and_reactive_names:
        :param rated_power:
        :param list generators: Generator name.
        :param list regulation_keys:
        :param reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """
        self.logger.info("Start dynamic power constrained regulation")

        if not rated_power:
            rated_power = self.data.get_rms_power(generators)

        active_and_reactive_names = []
        for index, i in enumerate(active_and_reactive_index):
            active_and_reactive_names.append([self.data.gtnet["output"][i[0]]])
            active_and_reactive_names[index].append(self.data.gtnet["output"][i[1]])

        for (gen, max_p, p_v_names) in zip(generators, rated_power, active_and_reactive_names):
            self.logger.debug(
                f"{gen} has a rated power of {max_p}, with active power name '{p_v_names[0]}' and reactive power '{p_v_names[1]}'.")
        gen_list = []
        current_value_list = []
        regulate = []
        for (generator, mn, initial_value) in zip(generators, min_max, initial_values):
            gens = []
            min_value = mn[0]
            max_value = mn[1]
            gens.append(generator)
            gens.append(initial_value)
            gens.append(min_value)
            gens.append(max_value)
            regulate.append(True)
            current_value_list.append(initial_value)
            gen_list.append(gens)
        if isinstance(step, list):
            for index, value in enumerate(step):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(step)
        if isinstance(delay, list):
            for index, value in enumerate(delay):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(delay)
        results = []

        first_res = self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[
            result_index]
        first_res = self.__dynamic_stabilisation(current_value_list, first_res)
        if reference_value - offset <= first_res <= reference_value + offset:
            self.logger.info("System already at reference point")
            return current_value_list

        results.append(first_res)

        direction = True
        if reference_value < first_res:
            direction = False
        while True in regulate:
            old_diff = 0
            for i, rated_power_i in enumerate(rated_power):
                active_value = self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[
                    active_and_reactive_index[i][0]]
                reactive_value = \
                    self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[
                        active_and_reactive_index[i][1]]
                power = abs(complex(active_value, reactive_value))
                current_diff = power - rated_power_i
                if regulate[i]:
                    if direction and current_diff < old_diff:
                        old_diff = current_diff
                        index = i
                    elif not direction and current_diff > old_diff:
                        old_diff = current_diff
                        gen_list[i][4] = -step
                        index = i
            gen = gen_list[index]
            _step = gen[4]
            delay = gen[5]
            if regulate[index]:
                current_value_list[index] = round(current_value_list[index] + _step, 6)
                current_res = \
                    self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types, delay)[
                        result_index]
                if delay == -1:
                    current_res = self.__dynamic_stabilisation(current_value_list, current_res)
                results.append(current_res)
                if reference_value - offset < current_res < reference_value + offset:
                    self.logger.info("Optimal point found")
                    result = ""
                    for ind, gen in enumerate(gen_list):
                        result += f"{generators[ind]}:  {self.data.gtnet['input'][ind]} changed from {gen[1]} to {current_value_list[ind]} \n "
                    self.logger.info(f"The following value has been adjusted: \n {result}")
                    return current_value_list

                if (results[-2] < reference_value < results[-1]) or (
                        results[-2] > reference_value > results[-1]
                ):
                    return current_value_list
                if gen[3] <= current_value_list[index] + _step or current_value_list[index] + _step <= gen[2]:
                    regulate[index] = False
                    return

        return logging.warning("Optimal value not found!")

    def current_constraint_regulation(self, generators, min_max, initial_values, current_index, result_index,
                                      reference_value=377, rated_power=None, u_ll_rms=None, step=0.001, offset=.1,
                                      delay=0, keep_reg=False, continious_regulation=False):
        """Regulate generators based on their RMS current.

        :param list generators: List of the generators to be regulated.
        :param list regulation_keys: The parameters to be adapted.
        :param str reference_key: The reference parameter to be checked.
        :param float reference_value: The value of the reference parameter to be reached.
        :param list rated_power: List of the rated power.
        :param list rms_current_names: List of the RMS slider names in the runtime.
        :param list voltages: List of the RMS Line-to-Line voltages.
        :param float_or_list step: Increase or decrease step.
        :param float_or_list offset: Acceptable offset to the reference value.
        :param float_or_list delay: Delay between setting and reading the value.
        :param bool keep_reg: If True, keep adjusting the regulation_keys to adapt the RMS currents.
        :return:
        """
        one_time_reg = True
        self.logger.info("Start dynamic regulation")

        if rated_power is None:
            rated_power = self.data.get_rms_power(generators)
        if u_ll_rms is None:
            u_ll_rms = self.data.get_rated_voltage(generators)

        rms_current_names = []
        for index, i in enumerate(current_index):
            rms_current_names.append([self.data.gtnet["output"][i[0]]])
            rms_current_names[index].append(self.data.gtnet["output"][i[1]])

        rated_current = calculate_rms_current(rated_power, u_ll_rms)
        if not continious_regulation:
            for (gen, max_i) in zip(generators, rated_current):
                self.logger.debug(
                    f"{gen} has an RMS current of {max_i} kA.")
        gen_list = []
        current_value_list = initial_values
        regulate = []
        for (generator, min_max_values, initial_value) in zip(generators, min_max, initial_values):
            gens = []
            min_value = min_max_values[0]
            max_value = min_max_values[1]
            gens.append(generator)
            gens.append(initial_value)
            gens.append(min_value)
            gens.append(max_value)
            regulate.append(True)
            gen_list.append(gens)
        if isinstance(step, list):
            for index, value in enumerate(step):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(step)
        if isinstance(delay, list):
            for index, value in enumerate(delay):
                gen_list[index].append(value)
        else:
            for gen in gen_list:
                gen.append(delay)
        while continious_regulation or one_time_reg:
            current_value_list = self.__current_constraint_regulation(generators, current_value_list, reference_value,
                                                                      offset,
                                                                      regulate, current_index, rated_current, gen_list,
                                                                      step,
                                                                      result_index, continious_regulation)
            one_time_reg = False

        if not keep_reg:
            return current_value_list

        self.logger.debug("Try to adjust the RMS currents.")
        k = 0
        while True:
            current_diff = []
            curr_rms_res = []
            for i, current_i in enumerate(current_index):
                current_rms = self.__calculate_rms(current_value_list, current_i)
                curr_rms_res.append(current_rms)
                current_diff.append(current_rms - rated_current[i])

            if all(val > 0 for val in current_diff) or all(val < 0 for val in current_diff):
                self.logger.debug("Cannot adjust the values anymore.")
                result = ""
                for ind, gen in enumerate(gen_list):
                    result += f"{generators[ind]}:  {self.data.gtnet['input'][ind]} changed from {gen[1]} to {current_value_list[ind]}, I = {curr_rms_res[ind]} kV \n "
                self.logger.info(f"Final values: \n {result}")
                return
            lowest_diff = 0
            highest_diff = 0
            for i, diff in enumerate(current_diff):
                if diff > highest_diff:
                    highest_diff = diff
                    high_index = i
                if diff < lowest_diff:
                    lowest_diff = diff
                    low_index = i

            current_value_list[high_index] -= step
            current_res = \
                self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types, delay)[
                    result_index]
            if delay == -1:
                current_res = self.__dynamic_stabilisation(current_value_list, current_res)
            k += 1
            if reference_value - offset < current_res < reference_value + offset or current_res > reference_value + offset:
                self.logger.info("Here")
                continue

            current_value_list[low_index] += step
            current_res = \
                self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types, delay)[
                    result_index]
            if delay == -1:
                current_res = self.__dynamic_stabilisation(current_value_list, current_res)

            k += 1

        return

    def __current_constraint_regulation(self, generators, current_value_list, reference_value,
                                        offset,
                                        regulate, current_indexes, max_current, gen_list, step, result_index,
                                        continious_regulation):
        """Regulate multiple generators specifically.

        :param u:
        :param rms_current_names:
        :param optimal_power:
        :param list generators: Generator name.
        :param list regulation_values:
        :param list reference_key:
        :param reference_value:
        :param step:
        :param offset:
        :param delay:
        :return:
        """

        results = []
        first_res = self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[
            result_index]
        first_res = self.__dynamic_stabilisation(current_value_list, first_res)
        if reference_value - offset <= first_res <= reference_value + offset:
            if not continious_regulation:
                self.logger.info("System already at reference point")
            return current_value_list
        results.append(first_res)
        k = 0

        direction = True
        if reference_value < first_res:
            direction = False
        if all(reg == False for reg in regulate):
            self.logger.warning("Cannot regulate any of the given generators")
            exit()

        while True in regulate:
            current_rms = []

            for i, current_i in enumerate(current_indexes):
                current_rms = self.__calculate_rms(current_value_list, current_i)
                current_rms.append(current_rms)
            if direction and regulate[i]:
                index = current_rms.index(min(current_rms))
            elif not direction and regulate[i]:
                gen_list[i][4] = -step
                index = current_rms.index(max(current_rms))

            gen = gen_list[index]
            _step = gen[4]
            delay = gen[5]

            if regulate[index]:
                current_value_list[index] = round(current_value_list[index] + _step, 6)
                current_res = \
                    self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types, delay)[
                        result_index]
                if delay == -1:
                    current_res = self.__dynamic_stabilisation(current_value_list, current_res)
                results.append(current_res)
                if reference_value - offset < current_res < reference_value + offset:
                    self.logger.info("Optimal point found")
                    result = ""
                    for ind, gen in enumerate(gen_list):
                        result += f"{generators[ind]}:  {self.data.gtnet['input'][ind]} changed from {gen[1]} to {current_value_list[ind]} \n "
                    self.logger.info(f"The following value has been adjusted: \n {result}")
                    return current_value_list

                if (results[-2] < reference_value < results[-1]) or (
                        results[-2] > reference_value > results[-1]
                ):
                    return current_value_list
                if gen[3] <= current_value_list[index] + _step or current_value_list[index] + _step <= gen[2]:
                    regulate[index] = False
                    return current_value_list
        logging.warning("Optimal value not found!")
        return current_value_list

    def __calculate_rms(self, current_value_list, currents):
        """

        :param list currents:
        :return:
        """
        i = 1
        for current in currents:
            i += self.simulation.set_value_and_read_output(current_value_list, self.data.unpacked_types)[
                     current] ** 2
        rms_current = math.sqrt(i / 3)
        return rms_current

    def save_results(self, values, load_indexes, p_ref_indexes):
        """Save results in the data base.

        :param list values: The values to be set, (generally the last value of the regulation should be put as parameter).
        :param list load_indexes: The indexes of the load active and reactive power.
        :param list p_ref_indexes: The indexes of the p ref of the generators.
        :return:
        """
        data = []
        res = self.simulation.set_value_and_read_output(values, self.data.unpacked_types)
        for load_index in load_indexes:
            data.append(res[load_index])
        for p_ref_index in p_ref_indexes:
            data.append(res[p_ref_index])
        self.database.add_data_to_table(self.table_name, data)

    def predictive_regulation(self, diff=5, first=True, regulate=False):
        """Predict the best grid parameter based on previous runs and optionally continue regulating

        :param int diff: Maximum difference between current and stored data.
        :param boo first: Grab first match if True.
        :param bool regulate: Continue regulation if set to True.
        """
        res = self.find_matches(diff, first)
        if res:
            self.simulation.set_value_and_read_output(res, self.data.unpacked_types)

    def find_matches(self, current_values, diff, first):
        """Find matches in the data base.

        :param list current_values: The current values iof the loads in the simualtion.
        :param int diff: Maximum difference between current and stored data.
        :param boo first: Grab first match if True.
        :return list of the matched data.
        :rtype list
        """
        headers = self.database.headers(self.table_name)[:self.load_index]
        database_data = self.database.read_data(self.table_name)
        old_difference = diff * (self.load_index + 1)
        data = 0
        for stored_data in database_data:
            if all(abs(stored_data[index] - current_values[index]) > diff for
                   index in range(self.load_index)):
                continue
            current_diff = sum(
                [abs(stored_data[index] - current_values[index]) > diff for
                 index in range(self.load_index)])
            if first:
                self.logger.info(
                    f"Found first matching values in the database, setting {self.database.headers(self.table_name)[self.load_index:]} to {stored_data[self.load_index:]}")
                return stored_data[self.load_index:]
            if current_diff < old_difference:
                old_difference = current_diff
                data = stored_data
        if old_difference == diff * (self.load_index + 1):
            self.logger.debug(f"No values found in {self.table_name}")
            return False
        self.logger.info(
            f"Found values in the database, setting {self.database.headers(self.table_name)[self.load_index:]} to {data[self.load_index:]}")
        return data[self.load_index:]
