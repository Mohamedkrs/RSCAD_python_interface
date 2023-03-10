# -*- coding: utf-8 -*-
"""Simulation that establishes a connection to the RSCAD environment."""

import logging
import os
import socket
import struct
import time

from logger import Logger

BUFFER_SIZE = 1024


# pylint: disable=line-too-long
class GTNETSimulation:
    """Class for simulating a GTNET SKT testcase."""

    def __init__(self, ip, port):
        """Connect to GTNET SKT module through TCP Socket.

        :param ip: GTNET SKT module ip address.
        :param port: GTNET SKT module port.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger())
        self.logger.addHandler(stream_handler)

        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((ip, port))
                self.socket.settimeout(5)
                self.logger.info(f"Connection successful to port {ip}:{port}")
                self.logger.debug("Simulation started successfully")
                break
            except socket.timeout:
                self.logger.warning(f"It was not possible to to connect to {ip}:{port}. Retrying in 5 seconds..")
                time.sleep(5)

    def read_value(self, unpacked_types, delay=0):
        """Read output via GTNET SKT Module. The GTNET SKT Module needs to send data in order to receive it. Please
        set the first parameter in the GTNET module to an arbitrary component that do not interfere with the simulation.

        :param str unpacked_types: type of received results: Example: float and int => "fi.
        :param int delay: wait x seconds and then read output.
        """

        send_data_str = struct.pack(f">i", 1)
        self.socket.send(send_data_str)
        if delay > 0:
            time.sleep(delay)
        received_data = self.socket.recv(BUFFER_SIZE)

        return struct.unpack(f">{unpacked_types}", received_data)

    def set_value_and_read_output(self, values: list, unpacked_types: str, delay=0):
        """When connecting to RSCAD through SKT module. Send parameter values and returns the given output.

        :param list values: values send to RSCAD. Need to be in the order as they are in the SKT mudule.
        :param str unpacked_types: type of received results: Example: float and int => "fi"
        :param float delay: The amount of seconds to wait between setting and reading the output.
        :return: output values in form of a list.
        """

        keys = ""
        for value in values:
            if isinstance(value, int):
                keys += "i "
            elif isinstance(value, float):
                keys += "f "
            elif isinstance(value, str):
                keys += f"{len(value)}s"
            else:
                print("Case not handled")
                exit()

        send_data_str = struct.pack(f">{keys}", *values)
        self.socket.send(send_data_str)

        if delay != 0:
            self.socket.recv(BUFFER_SIZE)
            time.sleep(delay)
            self.socket.send(send_data_str)
        try:
            received_data = self.socket.recv(BUFFER_SIZE)
        except socket.timeout:
            print("Could not receive data")
            return "please check that the send data are matching the GTNET module data"
        return struct.unpack(f">{unpacked_types}", received_data)


class RuntimeSimulation:
    """Class to simulate connection to RSCAD runtime."""

    def __init__(self, ip, port):
        """Initialize the simulation.

        :param str ip: Runtime ip address.
        :param int port: Runtime port number.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger())
        self.logger.addHandler(stream_handler)

        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((ip, port))
                self.socket.settimeout(2)
                self.logger.info(f"Connection successful to port {ip}:{port}")
                self.socket.send(b"Start;")
                data = b''
                timeout = 30
                while timeout > 0:
                    try:
                        part = self.socket.recv(BUFFER_SIZE)
                        data += part
                        if "courier is not in a valid download state" in data.decode("utf-8"):
                            self.logger.debug("Simulation already running")
                            return
                        elif "Successfully" in data.decode("utf-8"):
                            self.logger.debug("Simulation started successfully")
                            return
                    except socket.timeout:
                        pass
                    timeout -= 0
                    time.sleep(1)
            except socket.error:
                self.logger.warning(f"It was not possible to to connect to {ip}:{port}. Retrying in 5 seconds..")
                time.sleep(5)

    def stop(self):
        """Stop the simulation."""
        logging.info("Stopping the simulation")
        self.socket.send(b"stop;")

    def read_value(self, parameter, capture_type="MeterCapture", return_val=False):
        """Read the value from the runtime.

        :param str parameter: The parameter value to be read.
        :param str capture_type: the slider type. Default 'MeterCapture'.
        :param bool return_val: Set the return value as False if set to True instead of exiting the program. Default False.
        :return: The output value.
        :rtype float
        """
        self.socket.send(f'temp_float = {capture_type}("{parameter}");'.encode())
        self.socket.send(b"ListenOnPortHandshake(temp_float);")
        time.sleep(0.1)

        timeout = 10
        while timeout > 0:
            timeout -= 1
            try:
                self.socket.send(f'temp_float = {capture_type}("{parameter}");'.encode())
                self.socket.send(b"ListenOnPortHandshake(temp_float);")
                time.sleep(0.1)
                token_string = self.socket.recv(BUFFER_SIZE).decode("utf-8")

                if "cannot find a unique" in token_string:
                    self.logger.error(f"Cannot find a unique component with the name {parameter}")
                    if return_val:
                        return False
                    exit()
                else:
                    try:
                        return float(token_string.split()[0])
                    except ValueError:
                        pass
            except socket.timeout:
                self.logger.warning(f"Did not receive data for {parameter} in time, {timeout} retries left ")
            time.sleep(1)
        exit()

    def set_value(self, path_and_value, plot=None, plot_save_name=None):
        """Set a value in the Runtime.

        :param dict path_and_value: Dictionary containing the path and value for on or multiple components.
        :param str plot: Runtime plot name.
        :param str plot_save_name: Save name.
        :return: True if value is set correctly, False otherwise.
        :rtype bool.
        """
        for key, val in path_and_value.items():
            while True:
                try:
                    self.socket.send(f'SetSlider "{key}" = {val};'.encode())
                    time.sleep(1)
                    res = self.socket.recv(1024).decode("utf-8")

                    if "ERROR" in res:
                        self.logger.error(f"cannot find a unique slider with the path {key}")
                        return False
                    if plot is not None:
                        self.save_plot_as_csv(plot, f"{plot_save_name}")
                    return True
                except socket.timeout:
                    return True

    def set_switch_value(self, switches, set_value=0):
        """Set a value in the Runtime.

        :param str or list switches: The switches to be switch on or off.
        :param int set_value: Value to be set for the switches.
        :return: True if all command are successfully executed.
        """
        if isinstance(switches, str):
            switches = [switches]
        for switch in switches:
            try:
                self.socket.send(f'SetSwitch "{switch}" = {set_value};'.encode())
                time.sleep(1)
                res = self.socket.recv(1024).decode("utf-8")
                if "ERROR" in res:
                    logging.error(f"cannot find a unique switch with the name {switch}")
                    return False
            except socket.timeout:
                return True
        return True

    # pylint: disable=too-many-arguments
    def set_value_and_read_output(self, path_and_value, output, delay=0, plot=None, plot_save_name=None):
        """Set a value and read the output.

        :param dict path_and_value: A dictionary containing path and new value of a parameter.
        :param list output: A list containing the values to be read.
        :param float delay: The amount of seconds to wait between setting and reading the output.
        :param str plot: Runtime plot name.
        :param str plot_save_name: Save name.
        :return res: output values.
        :rtype float
        """
        if self.set_value(path_and_value, plot, plot_save_name):
            if output:
                if delay > 0:
                    time.sleep(delay)
                for out in output:
                    res = self.read_value(out)
                    return res
            return True
        else:
            return False

    def save_plot_as_csv(self, plot, save_name):
        """"Save a plot as a CSV file.

        :param str plot: The name of plot to save.
        :param str save_name: The name to save the plot as.
        """
        directory_path = os.getcwd()
        self.socket.send(f'SavePlotToCSV "{plot}", "{directory_path}\\{save_name}.csv";'.encode())

    def save_plot_as_jpeg(self, plot, save_name: str):
        """"Save a plot as a JPEG file.

        :param str plot: The name of plot to save.
        :param str save_name: The name to save the plot as.
        """
        directory_path = os.getcwd()
        self.socket.send(
            f'SavePlotToJpeg "{plot}", "{directory_path}\\{save_name}.jpeg",WIDTH,-1,HEIGHT,-1,UNITS,inches;'.encode())

    def save_plot_as_emf(self, plot, save_name: str):
        """"Save a plot as an EMF file.

        :param str plot: The name of plot to save.
        :param str save_name: The name to save the plot as.
        """
        directory_path = os.getcwd()
        self.socket.send(f'SavePlotToEMF "{plot}", "{directory_path}\\{save_name}.emf";'.encode())

    def create_data_frame(self, plot, save_name):
        """ Reads the graph in RSCAD and create .out file from which a csv file can be created.

        :param str plot: The name of plot to save.
        :param str save_name: The name to save the plot as.
        """
        directory_path = os.getcwd()
        self.socket.send(f'string data,dummy;'.encode())
        self.socket.send(f'fprintf(stdmsg,"Saving plot data for Case Number \n");'.encode())
        self.socket.send(f'SavePlot "{plot}","{directory_path}\\{save_name}.mpb";'.encode())
        self.socket.send(f'fscanf("{save_name}.out","%s%s%s%s%s",dummy,dummy,dummy,dummy,dummy);'.encode())
