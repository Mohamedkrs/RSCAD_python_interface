from regulate import RuntimeRegulation, GTNETRegulation
from simulation import GTNETSimulation, RuntimeSimulation
import data_extractor
TCP_IP = "127.0.0.1"
GTNET_SKT_IP = "10.203.156.61"
TCP_PORT = 4575

if __name__ == "__main__":
    regulator = GTNETRegulation(GTNET_SKT_IP, TCP_PORT,"9_bus")
    # initial_values = [0.043, 0.036]
    # continious_regulation = True
    # regulator.current_constraint_regulation(["BUS1x1", "BUS3x1"], min_max=[[0, 1.2], [0, 1.2]],
    #                                         initial_values=initial_values,
    #                                         current_index=[[1, 2, 3], [4, 5, 6]],
    #                                         result_index=0, delay=2,
    #                                         continious_regulation=continious_regulation)
    d = data_extractor.DataExtractor()
    print(d.get_database_dict_GTNET(["P_BUIS3x1","P_BUIS3x2", "P_BUS3x3"]))