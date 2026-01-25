from .Selector import Selector


class Validator(Selector):
	def __init__(self,
				 stage_name: str,
				 plotter,
				 markers,
				 clear_button: bool = False,
				 save_button: bool = False):
		buttons = [("accept", "Accept"), ("deny", "Deny")]
		super().__init__(stage_name, plotter, markers, buttons, clear_button, save_button)

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
