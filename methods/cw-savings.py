import vrplib

data = vrplib.read_instance('instances/A/A-n32-k5.vrp')

def cw_savings(data):
    # Initialise to a single node in each route
    routes = [[i] for i in range(1,data['dimension'])]

    # Calculates savings
    def saving(i,j):
        return data['edge_weight'][i][0]+data['edge_weight'][0][j]-data['edge_weight'][i][j]

    savings = []
    for i in range(2,data['dimension']):
        savings.append([saving(i,j) for j in range(1,i)])