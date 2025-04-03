from __future__ import annotations

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from pyctf import dsopen
# %%

DATA_DIR = '../../NaturalObject/MEG/MEG-BIDS'
RESULTS_DIR = '../../NOD-MEEG_results'
FONT_PATH = '../assets/Helvetica.ttc'
ELECTRODES = ['nas', 'lpa', 'rpa']
N_PARTICIPANTS = 30
SESSIONS_CONFIG = {
    range(1, 10): {
        'ses-ImageNet01': 2,
        'ses-ImageNet02': 2,
        'ses-ImageNet03': 8,
        'ses-ImageNet04': 8,
    },
    range(10, 31): {
        'ses-ImageNet01': 5,
    },
}

# %%


def analyze_head_movement(
    data_dir: str,
    results_dir: str,
    n_participants: int,
    sessions_config: dict[int, dict[str, int]],
    electrodes: list[str],
    font_path: str | None = None,
    palette: str = 'Spectral',
    fontsize: int = 12,
) -> None:
    """
    Analyze head movement by loading sensor positions, computing movements, and plotting results.

    Parameters
    ----------
    data_dir : str
        Root directory containing MEG data files(need CTF.ds data).
    results_dir : str
        Directory where results will be saved.
    n_participants : int
        Number of participants.
    sessions_config : Dict[int, Dict[str, int]]
        Dictionary mapping subjects ranges to their session configurations.
        Example: {range(1, 10): {'ses-ImageNet01': 2, 'ses-ImageNet02': 2}}
    electrodes : List[str]
        List of electrode names to consider.
    font_path : Optional[str], default=None
        Path to the font file to be used in plots.
    palette : str, default='Spectral'
        Color palette to use for plots.
    fontsize : int, default=16
        Font size to use in plots.

    Returns
    -------
    None
        The function saves plots and statistics to the specified directories.

    Notes
    -----
    This function encapsulates the process of loading MEG sensor data, computing head movement
    within and between sessions for multiple participants, and generating plots and statistics.
    It follows the open/closed principle by allowing extension through parameters without modifying
    the existing code structure.
    """

    # Set font and plot styles
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(
            fname=font_path,
        ).get_name()

    # Constants and settings
    palette_colors = sns.color_palette(palette=palette, n_colors=10)
    colors = [palette_colors[1], palette_colors[-2]]

    # Load sensor data
    data_loader = SensorDataLoader(
        root_dir=data_dir,
        n_participants=n_participants,
        sessions_config=sessions_config,
        electrodes=electrodes,
    )
    sensor_data = data_loader.load_sensor_positions()
    os.makedirs(f'{results_dir}/data', exist_ok=True)
    sensor_data.to_csv(f'{results_dir}/data/MEG-head_motion.csv', index=False)

    # Analyze head movement
    analyzer = HeadMovementAnalyzer(
        sensor_data=sensor_data,
        electrodes=electrodes,
        n_participants=n_participants,
        results_dir=results_dir,
        colors=colors,
        fontsize=fontsize,
    )
    analyzer.compute_head_movement()
    analyzer.plot_head_motion()
    analyzer.print_motion_statistics()


class SensorDataLoader:
    """
    Class to load and store sensor positions for participants.

    Parameters
    ----------
    root_dir : str
        Root directory containing MEG data files.
    n_participants : int
        Number of participants.
    sessions_config : Dict[int, Dict[str, int]]
        Dictionary mapping subjects ranges to their session configurations.
    electrodes : List[str]
        List of electrode names to consider.

    Attributes
    ----------
    root_dir : str
        Root directory containing MEG data files.
    n_participants : int
        Number of participants.
    sessions_config : Dict[int, Dict[str, int]]
        Session configurations for participants.
    electrodes : List[str]
        List of electrode names.
    columns : List[str]
        Column names for the sensor data DataFrame.
    data : pd.DataFrame
        DataFrame to store the sensor positions.
    """

    def __init__(
        self,
        root_dir: str,
        n_participants: int,
        sessions_config: dict[int, dict[str, int]],
        electrodes: list[str],
    ) -> None:
        self.root_dir = root_dir
        self.n_participants = n_participants
        self.sessions_config = sessions_config
        self.electrodes = electrodes
        self.columns = ['subjects', 'session', 'run'] + [
            f"{elec}_{axis}" for elec in electrodes for axis in ['x', 'y', 'z']
        ]
        self.data = pd.DataFrame(columns=self.columns)

    def load_sensor_positions(self) -> pd.DataFrame:
        """
        Load sensor positions for all participants, sessions, and runs.

        Returns
        -------
        pd.DataFrame
            DataFrame containing sensor positions for all runs.
        """
        all_rows: list[dict[str, Any]] = []
        for subjects in range(1, self.n_participants + 1):
            participant_sessions = self._get_participant_sessions(subjects)
            for session, n_runs in participant_sessions.items():
                for run in range(1, n_runs + 1):
                    row = self._load_single_run(subjects, session, run)
                    if row:
                        all_rows.append(row)
        self.data = pd.DataFrame(all_rows, columns=self.columns)
        return self.data

    def _get_participant_sessions(self, subjects: int) -> dict[str, int]:
        """
        Get the session configuration for a subjects.

        Parameters
        ----------
        subjects : int
            subjects number.

        Returns
        -------
        Dict[str, int]
            Session configuration for the subjects.
        """
        for participant_range, sessions in self.sessions_config.items():
            if subjects in participant_range:
                return sessions
        return {}

    def _load_single_run(
        self,
        subjects: int,
        session: str,
        run: int,
    ) -> dict[str, Any] | None:
        """
        Load sensor positions for a single run.

        Parameters
        ----------
        subjects : int
            subjects number.
        session : str
            Session identifier.
        run : int
            Run number.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing sensor positions for the run,
            or None if the file is not found.
        """
        meg_fn = (
            f"{self.root_dir}/sub-{subjects:02d}/{session}/meg/"
            f"sub-{subjects:02d}_{session}_task-{session[4:-2]}_run-{run:02}_meg.ds"
        )
        if os.path.exists(meg_fn):
            ds = dsopen(meg_fn)
            row: dict[str, Any] = {
                'subjects': subjects,
                'session': session,
                'run': run,
            }
            for i, elec in enumerate(self.electrodes):
                row.update({
                    f"{elec}_x": ds.dewar[i][0],
                    f"{elec}_y": ds.dewar[i][1],
                    f"{elec}_z": ds.dewar[i][2],
                })
            return row
        else:
            print(f"File not found: {meg_fn}")
            return None


class HeadMovementAnalyzer:
    """
    Class to compute and analyze head movement data.

    Parameters
    ----------
    sensor_data : pd.DataFrame
        DataFrame containing sensor positions.
    electrodes : List[str]
        List of electrode names.
    n_participants : int
        Number of participants.
    results_dir : str
        Directory where results will be saved.
    colors : List[Any]
        List of colors for plotting.
    fontsize : int
        Font size to use in plots.

    Attributes
    ----------
    sensor_data : pd.DataFrame
        DataFrame containing sensor positions.
    head_motion : pd.DataFrame
        DataFrame to store computed head motion data.
    electrodes : List[str]
        List of electrode names.
    n_participants : int
        Number of participants.
    results_dir : str
        Directory where results will be saved.
    colors : List[Any]
        List of colors for plotting.
    fontsize : int
        Font size to use in plots.
    """

    def __init__(
        self,
        sensor_data: pd.DataFrame,
        electrodes: list[str],
        n_participants: int,
        results_dir: str,
        colors: list[Any],
        fontsize: int,
    ) -> None:
        self.sensor_data = sensor_data
        self.head_motion = pd.DataFrame()
        self.electrodes = electrodes
        self.n_participants = n_participants
        self.results_dir = results_dir
        self.colors = colors
        self.fontsize = fontsize

    @staticmethod
    def calculate_distance(
        coord1: np.ndarray,
        coord2: np.ndarray,
    ) -> float:
        """
        Calculate Euclidean distance between two 3D coordinates.

        Parameters
        ----------
        coord1 : np.ndarray
            First coordinate (3D vector).
        coord2 : np.ndarray
            Second coordinate (3D vector).

        Returns
        -------
        float
            Euclidean distance between the two coordinates.
        """
        return np.linalg.norm(coord1 - coord2)

    def compute_head_movement(self) -> pd.DataFrame:
        """
        Compute head movement within and between sessions for all participants.

        Returns
        -------
        pd.DataFrame
            DataFrame containing head movement data.
        """
        participants = self.sensor_data['subjects'].unique()
        motion_records: list[dict[str, Any]] = []

        for subjects in participants:
            participant_data = self.sensor_data[self.sensor_data['subjects'] == subjects]
            sessions = participant_data['session'].unique()

            # Within-session head movement
            motion_records.extend(
                self._compute_within_session_movement(
                    participant_data, subjects,
                ),
            )

            # Between-session head movement
            motion_records.extend(
                self._compute_between_session_movement(
                    participant_data, subjects, sessions,
                ),
            )

        self.head_motion = pd.DataFrame(motion_records)
        return self.head_motion

    def _compute_within_session_movement(
        self,
        participant_data: pd.DataFrame,
        subjects: int,
    ) -> list[dict[str, Any]]:
        """
        Compute within-session head movement for a subjects.

        Parameters
        ----------
        participant_data : pd.DataFrame
            Sensor data for the subjects.
        subjects : int
            subjects number.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing within-session motion records.
        """
        motion_records: list[dict[str, Any]] = []
        sessions = participant_data['session'].unique()
        for session in sessions:
            session_data = participant_data[participant_data['session'] == session]
            runs = session_data['run'].values
            coords = self._extract_coordinates(session_data)

            for i in range(len(runs) - 1):
                avg_motion = self._compute_average_motion(coords, i, i + 1)
                motion_records.append({
                    'subjects': subjects,
                    'session': session,
                    'run': runs[i + 1],
                    'motion': avg_motion,
                    'type': 'within-session',
                })
        return motion_records

    def _compute_between_session_movement(
        self,
        participant_data: pd.DataFrame,
        subjects: int,
        sessions: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Compute between-session head movement for a subjects.

        Parameters
        ----------
        participant_data : pd.DataFrame
            Sensor data for the subjects.
        subjects : int
            subjects number.
        sessions : np.ndarray
            Array of session identifiers.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing between-session motion records.
        """
        motion_records: list[dict[str, Any]] = []
        if len(sessions) > 1:
            for i in range(len(sessions) - 1):
                session1 = sessions[i]
                session2 = sessions[i + 1]
                data1 = participant_data[participant_data['session'] == session1]
                data2 = participant_data[participant_data['session'] == session2]
                coords1 = self._extract_coordinates(data1)
                coords2 = self._extract_coordinates(data2)

                for idx1, run1 in enumerate(data1['run']):
                    for idx2, run2 in enumerate(data2['run']):
                        avg_motion = self._compute_average_motion(
                            coords1, idx1, idx2=idx2, coords2=coords2,
                        )
                        motion_records.append({
                            'subjects': subjects,
                            'session1': session1,
                            'session2': session2,
                            'run1': run1,
                            'run2': run2,
                            'motion': avg_motion,
                            'type': 'between-session',
                        })
        return motion_records

    def _extract_coordinates(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Extract coordinates for electrodes from the data.

        Parameters
        ----------
        data : pd.DataFrame
            Sensor data.

        Returns
        -------
        np.ndarray
            Numpy array of shape (n_runs, n_electrodes, 3) containing coordinates.
        """
        coord_columns = [
            f"{elec}_{axis}" for elec in self.electrodes for axis in ['x', 'y', 'z']
        ]
        return data[coord_columns].values.reshape(-1, len(self.electrodes), 3)

    def _compute_average_motion(
        self,
        coords1: np.ndarray,
        idx1: int,
        idx2: int | None = None,
        coords2: np.ndarray | None = None,
    ) -> float:
        """
        Compute average motion between two sets of coordinates.

        Parameters
        ----------
        coords1 : np.ndarray
            Coordinates from the first data set.
        idx1 : int
            Index of the first coordinate set.
        idx2 : Optional[int], default=None
            Index of the second coordinate set.
        coords2 : Optional[np.ndarray], default=None
            Coordinates from the second data set.

        Returns
        -------
        float
            Average motion between the two coordinate sets.
        """
        if coords2 is None:
            coords2 = coords1
            idx2 = idx2 if idx2 is not None else idx1 + 1
        distances = [
            self.calculate_distance(coords1[idx1][j], coords2[idx2][j]) for j in range(len(self.electrodes))
        ]
        return np.mean(distances)

    def plot_head_motion(self) -> None:
        """
        Plot head motion using violin plots for within and between session movements.

        Returns
        -------
        None
            The function saves the plot to the specified results directory.
        """
        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
        sns.violinplot(
            data=self.head_motion,
            x='subjects', y='motion', hue='type',
            inner='point', inner_kws={'s': 0.5, 'color': 'grey', 'alpha': 0.4},
            ax=ax, native_scale=True,
            linewidth=0, density_norm='width',
            palette=self.colors, legend=False,
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Subjects', fontsize=self.fontsize)
        ax.set_ylabel('Head Motion (mm)', fontsize=self.fontsize)
        ax.set_xticks(np.arange(1, self.n_participants + 1))
        ax.legend(
            handles=[
                Patch(color=self.colors[1], label='Between Session'),
                Patch(color=self.colors[0], label='Within Session'),
            ], loc='upper right',
        )
        plt.tight_layout()
        os.makedirs(f'{self.results_dir}/figs', exist_ok=True)
        plt.savefig(
            f'{self.results_dir}/figs/head_motion.svg',
            dpi=600, bbox_inches='tight',
        )
        plt.show()

    def print_motion_statistics(self) -> None:
        """
        Calculate and print median head motion statistics.

        Returns
        -------
        None
            The function prints the statistics to the console.
        """
        within_motion = self.head_motion[
            self.head_motion['type']
            == 'within-session'
        ]
        between_motion = self.head_motion[
            self.head_motion['type']
            == 'between-session'
        ]

        within_median = within_motion.groupby(
            'subjects',
        )['motion'].mean().median()
        between_median = between_motion.groupby(
            'subjects',
        )['motion'].mean().median()

        print(f'Within-session median: {within_median:.3f} mm')
        print(f'Between-session median: {between_median:.3f} mm')


# %%
analyze_head_movement(
    data_dir=DATA_DIR,
    results_dir=RESULTS_DIR,
    n_participants=N_PARTICIPANTS,
    sessions_config=SESSIONS_CONFIG,
    electrodes=ELECTRODES,
    font_path=FONT_PATH,
)
