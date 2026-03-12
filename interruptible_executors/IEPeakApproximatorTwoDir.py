from .IEPeakApproximator import IEPeakApproximator


class IEPeakApproximatorTwoDir(IEPeakApproximator):
    """
    IEPeakApproximator with exactly two trajectories:
    - Trajectory 1 is processed in increasing field direction.
    - Trajectory 2 is processed in decreasing field direction.
    """

    def __init__(self, data: dict, stage_name: str, initial_params: dict = None, save_figure_path: str = None):
        super().__init__(data, stage_name, initial_params, save_figure_path)

        if len(self.input_trajectories) != 2:
            raise ValueError("IEPeakApproximatorTwoDir requires exactly 2 trajectories")

        self._apply_two_dir_order()

    def _apply_two_dir_order(self):
        """
        Переворачивает вторую траекторию, чтобы обработка шла с конца.
        """
        traj = self.input_trajectories[1]

        reversed_traj = {
            "fields": list(reversed(traj.get("fields", []))),
            "freq": list(reversed(traj.get("freq", []))),
            "width": list(reversed(traj.get("width", []))),
            "magnitude": list(reversed(traj.get("magnitude", []))),
            "prominence": list(reversed(traj.get("prominence", [])))
        }

        for key, value in traj.items():
            if key not in reversed_traj:
                reversed_traj[key] = value

        self.input_trajectories[1] = reversed_traj