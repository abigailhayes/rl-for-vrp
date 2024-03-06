import instances.utils as instance_utils
import methods.cw_savings as cw_savings

data = instance_utils.import_instance('A', 'A-n32-k5')

test = cw_savings.CWSavings(data['instance'])
test.add_sol(data['solution'])
test.run_all()
print(test.cost, " Perc worse: ", '{:.1%}'.format(test.perc))