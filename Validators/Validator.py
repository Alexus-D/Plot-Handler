from plotters import Plotter

from selectors import Selector


class Validator(Selector):
	def __init__(self,
				 stage_name: str,
				 plot_data: dict,
				 clear_button: bool = False,
				 save_button: bool = False):
		buttons = [("accept", "Accept"), ("deny", "Deny")]
		super().__init__(stage_name, plot_data, buttons, clear_button, save_button)

	def _set_mode(self, mode):
		if mode == "accept":
			self.params = {"Validation": True}
			self.mode = None
			return
		if mode == "deny":
			self.params = {"Validation": False}
			self.mode = None
			return
		raise ValueError(f"Mode {mode} is not recognized")

	def mouse_on_click(self, event):
		return
	
	def _create_plotter(self, plot_data: dict) -> Plotter:
		raise NotImplementedError("_create_plotter must be implemented in Validator subclass")
	
	def _create_markers(self, plotter: Plotter):
		return {}