from executors import ECut, EPeakParams, ETrajectories

from .IEPeakWatcher import IEPeakWatcher

###LEGACY: IEPeakWatcherTwoDir - для двух траекторий с жестким направлением. Сейчас не используется, но может пригодиться для сравнения с IEPeakWatcher.
class IEPeakWatcherTwoDir(IEPeakWatcher):
    """
    IEPeakWatcher with exactly two trajectories:
    - Trajectory 1 is tracked in increasing field direction.
    - Trajectory 2 is tracked in decreasing field direction.
    """

    def select_initial_params(self):
        """
        1) Выбор траекторий
        2) Для 1-й траектории: параметры пика в начале
        3) Для 2-й траектории: параметры пика в конце
        """
        print("[IEPeakWatcherTwoDir] Старт: выбор начальных параметров")

        traj_executor = ETrajectories(self.data, "Select Trajectories")
        traj_executor.execute_with_validation()
        self.trajectories = traj_executor.get_result()["trajectories"]

        if len(self.trajectories) != 2:
            raise ValueError("IEPeakWatcherTwoDir requires exactly 2 trajectories")

        self.initial_peak_params = []

        for i, (start, end) in enumerate(self.trajectories):
            # Для второй траектории выбираем точку в конце
            anchor_point = start if i == 0 else end
            anchor_field = anchor_point[0]
            anchor_freq = anchor_point[1]

            print(f"[IEPeakWatcherTwoDir] Траектория {i+1}: срез в поле {anchor_field}")

            cut_executor = ECut(
                self.data,
                f"Cut for trajectory {i+1}",
                axis="x",
                initial_params={"cut_value": anchor_field}
            )
            cut_executor.execute()
            cut_data = cut_executor.get_result()

            cut_data["highlight_points"] = [
                (anchor_freq, self._get_z_at_point(anchor_field, anchor_freq))
            ]

            peak_executor = EPeakParams(cut_data, f"Peak params for trajectory {i+1}")
            print(f"[IEPeakWatcherTwoDir] Траектория {i+1}: выбор параметров пика")
            peak_executor.execute_with_validation()
            peak_params = peak_executor.get_result()

            self.initial_peak_params.append(peak_params)

        self.result = {"trajectories": []}
        print("[IEPeakWatcherTwoDir] Готово: начальные параметры выбраны")

    def _get_fields_range(self, start_field, end_field):
        """
        Возвращает список индексов полей от start_field до end_field,
        принудительно задавая направление для двух траекторий.
        """
        fields = self.data["x"]

        start_idx = (abs(fields - start_field)).argmin()
        end_idx = (abs(fields - end_field)).argmin()

        # Первая траектория - вверх по полю, вторая - вниз
        direction_up = (self._current_traj_idx == 0)

        if direction_up:
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            indices = list(range(start_idx, end_idx + 1))
        else:
            if start_idx < end_idx:
                start_idx, end_idx = end_idx, start_idx
            indices = list(range(start_idx, end_idx - 1, -1))

        return indices