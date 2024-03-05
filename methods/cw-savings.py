import vrplib

data = vrplib.read_instance('instances/A/A-n32-k5.vrp')

def cw_savings(data):
    # Initialise to a single node in each route
    routes = [[i] for i in range(1,data['dimension'])]

