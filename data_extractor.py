# -*- coding: utf-8 -*-
"""Class that collects data from grid."""

import logging
import os.path
import re

import yaml

from logger import Logger


# pylint: disable=line-too-long
class DataExtractor:
    """Extract grid data from given files."""

    def __init__(self, inf_file=None, dtp_file=None, sib_file=None):
        """Initialize DataExtractor class.

        :param: str inf_file: path to inf file.
        :param: str dtp_file: path to dtp file.
        :param: str sib_file: path to sib file.
        """
        self.logger = logging.getLogger("Data Extractor")
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger())
        self.logger.addHandler(stream_handler)

        self.slider_names = {}
        self.unpacked_types = ""
        self.packed_types = ""
        self.gtnet = {"input": [], "output": []}
        self.grid_data = {"generators": {}, "loads": {}, "transmission_lines": {}}
        if inf_file is None:
            for i in os.listdir("."):
                if i.endswith(".inf"):
                    self.__inf_file = i
        else:
            self.__inf_file = inf_file

        if dtp_file is None:
            for i in os.listdir("."):
                if i.endswith(".dtp"):
                    self.__dtp_file = i
        else:
            self.__dtp_file = dtp_file
        if sib_file is None:
            for i in os.listdir("."):
                if i.endswith(".sib"):
                    self.__sib_file = i
        else:
            self.__sib_file = sib_file
        if not os.path.exists(self.__inf_file) or not os.path.exists(self.__dtp_file) or not os.path.exists(
                self.__sib_file):
            self.logger.error("Cannot find .dtp or .inf file. Please make sure that they exist in the current folder")
            exit()

        self.collect_data()

    def __str__(self):
        """Print grid_data in a readable way."""
        return yaml.dump(self.grid_data, indent=4, default_flow_style=False)

    def filter_data(self, search_word, nested_dict=None):
        """Return grid_data of containing certain search_word.

        :param dict nested_dict: The dictionary
        :param str search_word: The parameter, which data are displayed.
        :return Dictionary containing the filtered parameter data.
        :rtype dict
        """
        if nested_dict is None:
            nested_dict = self.grid_data

        result = {}
        for key, value in nested_dict.items():
            if search_word in key:
                result[key] = value
            elif isinstance(value, dict):
                nested_result = self.filter_data(search_word, value)
                if nested_result:
                    result[key] = nested_result
        return result

    def collect_data(self):
        """Collect data from inf and dtp files.

        :return: grid data.
        :rtype: dict
        """
        try:
            self.find_components()
            # self.find_components_dfx()
            self.assign_properties()
            self.gtnet_skt()
            self.get_slider_name()
        except KeyError:
            pass

        return self.grid_data

    def assign_properties(self):
        """Assign properties to generators, loads and transmission lines."""
        for value in self.grid_data.values():
            for key in value.keys():
                properties = self.component_data(key)
                if properties:
                    value[key].update(self.component_data(key))

    def find_components_dfx(self):
        """Find generators, loads and lines in dtp file."""

        with open("IEEE 9 Bus Power System.dfx", "r") as file:
            data = file.read()
            data = data.split("COMPONENT_TYPE")

            for comp in data:
                if "Generator" in comp:
                    generator_attributes = comp.split("\n")
                    for attribute in generator_attributes:
                        if "\t" in attribute and ":" in attribute:
                            attrib = attribute.replace("\t", "").split(":")
                            if len(attrib) == 2 and attrib[1] != "":
                                if "Name" in attrib:
                                    name = attrib[1]
                                    self.grid_data["generators"][name] = {}
                                else:
                                    self.grid_data["generators"][name][attrib[0]] = attrib[1]

    def find_components(self):
        """Find generators, loads and lines in dtp file."""

        with open(self.__dtp_file, "r") as file:
            data = file.read()
            data = data.split("RTDS.PSYS.RISC_CMODEL")

            for comp in data:
                if "generator" in comp:
                    gen_name = comp.split(" ")
                    gen_name = gen_name[gen_name.index("NAME:") + 1][:-1]
                    self.grid_data["generators"][gen_name] = {}
                    for line in comp.splitlines():
                        _line = line.split(" ")
                        if "mmva" in line:
                            self.grid_data["generators"][gen_name]["mmva"] = float(_line[_line.index("mmva") - 1])
                        if "Vbsll" in line:
                            self.grid_data["generators"][gen_name]["Vbsll"] = float(_line[_line.index("Vbsll") - 1])
                elif "_dyLoad111" in comp:
                    if "_dyLoad111" in comp:
                        load_name = comp.split(" ")
                        load_name = load_name[load_name.index("NameParam") + 2]
                        self.grid_data["loads"][load_name] = {}
                elif "TLterm" in comp:
                    line_data = comp.split("Tnam1")[0].split(" ")[-1]
                    line_name = line_data.strip("<(>")
                    line_data = comp.split("LENGTH")[0].strip().split(" ")[-1]
                    line_length = float(line_data)
                    line_data = comp.split("PERCENT_OF_LINE")[0].strip().split(" ")[-1]
                    line_percent = float(line_data)
                    if line_name not in self.grid_data["transmission_lines"].keys():
                        self.grid_data["transmission_lines"][line_name] = {
                            "length": line_length,
                            "percent": line_percent,
                        }

    def draft_data(self):
        """Collect draft data."""
        with open(self.__dtp_file, "r") as file:
            data = file.read()
            data = data.split("RTDS.PSYS.RISC_CMODEL")
            for gen in self.grid_data["generators"]:
                for comp in data:
                    if f"NameParam = {gen}" in comp:
                        print(comp)

    def component_data(self, component_name: str):
        """Find component data in inf file.

        :args: str component_name: name of component.
        :return: component.
        :rtype: dict
        """
        comp_data = {}
        with open(self.__inf_file, "r") as file:
            data = file.readlines()
            for line in data:
                if component_name in line and "Group" in line:
                    path = line[line.index("Group") + 7:]
                    path = path[: path.index('"')]
                    comp_data["path"] = path.replace("|", ":")
                    # path = line.split("=")
                    # if "Group" in path:
                    #     comp_data["Path"] = data[data.index(path)+1]
                    line = line.split(" ")
                    for param in line:
                        if "Rack" not in param and "Adr" not in param and "=" in param:
                            if "Desc" in param:
                                component = param.split("=")[1].strip('"')
                                comp_data[param.split("=")[1].strip('"')] = {}
                            else:
                                try:
                                    val = float(param.split("=")[1].strip('"'))
                                    comp_data[component][
                                        param.split("=")[0].strip('"')
                                    ] = val
                                except:
                                    if "Group" in param:
                                        comp_data[component]["path"] = (param.split("=")[1] + " " + line[
                                            line.index(param) + 1].replace("|", ":")).strip('"') + ":" + component
                                    comp_data[component][
                                        param.split("=")[0].strip('"')
                                    ] = param.split("=")[1].strip('"')
        return comp_data

    def get_slider_name(self):
        """Collect slider names."""
        with open(self.__sib_file, "r") as file:
            data = file.read()
            data = data.split("COMPONENT")
            load_active_reactive_power = self.load_active_reactive_power()
            generator_active_reactive_power = self.generator_active_reactive_power()
            for comp in self.grid_data.values():

                for key in comp.keys():
                    for component in data:
                        if "RMS" in component and key in component:
                            split_comp = component.split()
                            self.grid_data["generators"][key]["RMS"] = split_comp[split_comp.index("NAME:") + 1]
                            continue
                        if "SLIDER" in component and key in component:
                            for active_reactive_power in load_active_reactive_power:
                                for active_power in active_reactive_power:
                                    if key in component and active_power in component:
                                        name = component.split()
                                        name = name[name.index("NAME:") + 1]
                                        if name == "ICON:":
                                            name = ""
                                        comp[key][active_power]["name"] = name
                                        self.slider_names[key] = name
                        if "METER" in component and key in component:
                            for active_reactive_power in generator_active_reactive_power:
                                for active_power in active_reactive_power:
                                    if key in component and active_power in component:
                                        name = component.split()
                                        name = name[name.index("NAME:") + 1]
                                        if name == "ICON:":
                                            name = ""
                                        comp[key][active_power]["name"] = name
                                        self.slider_names[key] = name

    def find_loads(self):
        """Collect loads from dtp file.

        :args: str dtp_file: path to dtp file.
        :return: loads.
        :rtype: list
        """
        loads = []
        with open(self.__dtp_file, "r") as file:
            data = file.read()
            data = data.split("RTDS.PSYS.RISC_CMODEL")
            for load in data:
                if "_dyLoad111" in load:
                    load_name = load.split(" ")
                    load_name = load_name[load_name.index("NameParam") + 2]
                    loads.append(load_name)
        return loads

    def find_lines(self):
        """Collect transmission lines from dtp file.

        :args: str dtp_file: path to dtp file.
        :return: dictionary of transmission lines data.
        :rtype: dict
        """
        lines = {}
        with open(self.__dtp_file, "r") as file:
            data = file.read()
            data = data.split("RTDS.PSYS.RISC_CMODEL")
            for dt in data:
                if "TLterm" in dt:
                    line_data = dt.split("Tnam1")[0].split(" ")[-1]
                    line_name = line_data.strip("<(>")
                    line_data = dt.split("LENGTH")[0].strip().split(" ")[-1]

                    line_length = float(line_data)
                    line_data = dt.split("PERCENT_OF_LINE")[0].strip().split(" ")[-1]
                    line_percent = float(line_data)
                    if line_name not in lines.keys():
                        lines[line_name] = {"length": line_length, "percent": line_percent}
        return lines

    def find_generators(self):
        """Collect generators from dtp file.

        :args: str dtp_file: path to dtp file.
        :return: generators.
        :rtype: list
        """
        generators = []
        with open(self.__dtp_file, "r") as file:
            data = file.read()
            data = data.split("RTDS.PSYS.RISC_CMODEL")
            for gen in data:
                if "generator" in gen:
                    gen_name = gen.split(" ")
                    gen_name = gen_name[gen_name.index("NAME:") + 1][:-1]
                    generators.append(gen_name)
        return generators

    def find_gen_pv_data(self, generators=None):
        """Find PV data for given generator.

        :args: str gen_name: name of generator."""
        pv = []
        if generators is None:
            generators = list(self.grid_data["generators"].keys())
        if isinstance(generators, list):
            for gen in generators:
                gen = self.component_data(gen)
                for key, val in gen.items():
                    if isinstance(val, dict) and "P" in key and "Inputs" in val['path']:
                        pv.append(key)
            return pv
        else:
            gen = self.component_data(generators)
            for key, val in gen.items():
                if isinstance(val, dict) and "P" in key and "Inputs" in val['path']:
                    return key

    def gtnet_skt(self):
        """Extract GTNET values from .dtp file."""
        with open(self.__dtp_file, "r") as file:
            data = file.read()
            data = data.split("RTDS.CTL.RISC_CMODEL")
            for dt in data:
                if "GTNETSKT" in dt:
                    dt = dt.split(" ")
                    name = dt[dt.index("ModelName") + 2]
                    from_gtnet = dt[dt.index("numVarsToGTNETSKT") - 1]
                    to_gtnet = dt[dt.index("numVarsFromGTNETSKT") - 1]
                    if "GTNET" not in self.grid_data.keys():
                        self.grid_data["GTNET"] = {}
                    if name not in self.grid_data.keys():
                        self.grid_data["GTNET"][name] = {}
                    self.grid_data["GTNET"][name]["numVarsToGTNETSKT"] = from_gtnet
                    self.grid_data["GTNET"][name]["numVarsFromGTNETSKT"] = to_gtnet
                    for out in range(int(from_gtnet)):
                        for out_val in dt:
                            if f"out{out}x" in out_val:
                                if f"out{out}x" not in self.grid_data["GTNET"][name]:
                                    self.grid_data["GTNET"][name][f"out{out}x"] = {}
                                param_name = out_val[3:out_val.index(">")]
                                self.grid_data["GTNET"][name][f"out{out}x"]["name"] = param_name
                                self.gtnet["output"].append(param_name)
                                _type = self.get_type(param_name)
                                self.unpacked_types += "i " if _type == "INT" else "f "
                                self.grid_data["GTNET"][name][f"out{out}x"]["type"] = _type
                    for out in range(int(to_gtnet)):
                        for in_val in dt:
                            if f"in{out}x" in in_val:
                                if f"in{out}x" not in self.grid_data["GTNET"][name]:
                                    self.grid_data["GTNET"][name][f"in{out}x"] = {}
                                param_name = in_val[3:in_val.index(">")]
                                self.grid_data["GTNET"][name][f"in{out}x"]["name"] = param_name
                                self.gtnet["input"].append(param_name)
                                _type = self.get_type(param_name)
                                self.packed_types += "i " if _type == "INT" else "f "
                                self.grid_data["GTNET"][name][f"in{out}x"]["type"] = _type

    def get_type(self, param):
        """Get GTNET parameter type."""
        with open(self.__inf_file, "r") as file:
            data = file.readlines()
            for line in data:
                if "GTNETSKT" in line and param in line:
                    line = re.split(" |=", line)
                    return line[line.index("Type") + 1]
                if f'Output Desc="{param}"' in line:
                    line = re.split(" |=", line)
                    return line[line.index("Type") + 1]

    def get_rms_power(self, generators):
        """Get the generators RMS power."""
        max_power = []
        for gen in generators:
            max_power.append(self.grid_data["generators"][gen]["mmva"])
        return max_power

    def get_rms_currents(self, generators):
        """Get the RMS currents."""
        currents = []
        for gen in generators:
            try:
                currents.append(self.grid_data["generators"][gen]["RMS"])
            except KeyError as error:
                self.logger.error(f"No {error} currents detected")
                exit()
        return currents

    def get_rated_voltage(self, generators):
        """Get generator RMS voltages"""
        voltages = []
        for gen in generators:
            voltages.append(self.grid_data["generators"][gen]["Vbsll"])
        return voltages

    def get_generator_powers_name(self, generators=None):
        power = []
        if generators is None:
            generators = list(self.grid_data["generators"].keys())
        for gen in generators:
            active_reactive = []
            for param in self.grid_data["generators"][gen]:
                if isinstance(self.grid_data["generators"][gen][param], dict):
                    if "MW" in self.grid_data["generators"][gen][param].values() or "MVAR" in \
                            self.grid_data["generators"][gen][param].values():
                        if self.grid_data["generators"][gen][param]["name"] != "":
                            active_reactive.append(self.grid_data["generators"][gen][param]["name"])
                        else:
                            active_reactive.append(param)
                if len(active_reactive) == 2:
                    break
            power.append(active_reactive)
        return power

    def generator_active_reactive_power(self, generators=None):
        """Get load active and reactive power values.

        :param list generators: List of loads. All loads are included if no parameter is given."""
        active_reactive_power = []
        if generators is None:
            generators = self.grid_data["generators"].keys()
        for load in generators:
            power = []
            for param in self.grid_data["generators"][load]:
                if isinstance(self.grid_data["generators"][load][param], dict):
                    if "MW" in self.grid_data["generators"][load][param].values():
                        power.append(param)
                    if "MVAR" in self.grid_data["generators"][load][param].values():
                        power.append(param)
                if len(power) == 2:
                    break
            active_reactive_power.append(power)
        return active_reactive_power

    def load_active_reactive_power(self, loads=None):
        """Get load active and reactive power values.

        :param list loads: List of loads. All loads are included if no parameter is given."""
        active_reactive_power = []
        if loads is None:
            loads = self.grid_data["loads"].keys()
        for load in loads:
            power = []
            for param in self.grid_data["loads"][load]:
                if isinstance(self.grid_data["loads"][load][param], dict):
                    if "MW" in self.grid_data["loads"][load][param].values():
                        power.append(param)
                    if "MVAR" in self.grid_data["loads"][load][param].values():
                        power.append(param)
                if len(power) == 2:
                    break
            active_reactive_power.append(power)
        return active_reactive_power

    def get_database_dict(self):
        """Grab stored grid data."""
        database_dict = {}
        try:
            for load, active_reactive_power in zip(self.grid_data["loads"].values(), self.load_active_reactive_power()):
                for active_power in active_reactive_power:
                    database_dict[load[active_power]["name"]] = "FLOAT"
        except:
            self.logger.warning("Some components do not have a unique name")
        load_index = len(database_dict.keys())
        try:
            for load, active_reactive_power in zip(self.grid_data["generators"].values(), self.find_gen_pv_data()):
                database_dict[active_reactive_power] = "FLOAT"
        except:
            self.logger.warning("Some components do not have a unique name")
        return load_index, database_dict

    def get_database_dict_GTNET(self, gen_pref_names):
        """Grab stored grid data."""
        database_dict = {}
        try:
            for load, active_reactive_power in zip(self.grid_data["loads"].values(), self.load_active_reactive_power()):
                for active_power in active_reactive_power:
                    database_dict[load[active_power]["name"]] = "FLOAT"
        except:
            self.logger.warning("Some components do not have a unique name")
        load_index = len(database_dict.keys())
        for p_ref in gen_pref_names:
            database_dict[p_ref] = "FLOAT"
        return load_index, database_dict
