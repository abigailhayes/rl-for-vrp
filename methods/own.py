"""Code for using the sweep method, and any further methods that might be developed."""

from methods.cw_savings import CWSavings


class Own:

    def __init__(self, instance, method_settings):
        self.routes = None
        self.cost = None
        self.init_method = method_settings['init_method']
        self.improve_method = method_settings['improve_method']

        if self.init_method == 'savings':
            self.init_model = CWSavings(instance)

    def run_all(self):
        self.init_model.run_all()
        self.cost = self.init_model.cost
        self.routes = self.init_model.routes
        if self.improve_method != 'None':
            self.improve_model.run_all()
            self.cost = self.improve_model.cost
            self.routes = self.improve_model.routes
