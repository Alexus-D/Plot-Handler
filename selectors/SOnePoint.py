from .Selector import Selector
from .click_functions.Choose_point import choose_one_point

class SOnePoint(Selector):
    def __init__(self,
                 stage_name: str,
                 plot_data: dict = None,
                 buttons: list = None):
        if buttons is None:
            buttons = [("select_one_point", "Resonantor freq")]
        
        super().__init__(stage_name, plot_data, buttons)
    
    def set_click_funks(self):
        return {
            "select_one_point": choose_one_point
        }
    
    def redraw(self):
        pass # TODO: implement!!!