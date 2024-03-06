import vrplib
import methods.cw_savings as cw_savings

data = vrplib.read_instance('instances/A/A-n32-k5.vrp')
solution = vrplib.read_solution('instances/A/A-n32-k5.sol')

test = cw_savings.CWSavings(data)
test.add_sol(solution)
test.run_all()
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))