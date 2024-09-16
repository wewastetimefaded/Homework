import numpy as np
import matplotlib.pyplot as plt
from iapws_model import IapwsModel
from typing import List, Tuple, Optional, Union

def legend_without_duplicate_labels(ax: plt.Axes) -> None:
     handles, labels = ax.get_legend_handles_labels()
     unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
     ax.legend(*zip(*unique))

def plot_process(ax: plt.Axes, points: List[IapwsModel], **kwargs) -> None:
    ax.plot([point.s for point in points], [point.h for point in points], **kwargs)

def get_isobar(point: IapwsModel) -> Tuple[List[float], List[float]]:
    s = point.s
    s_values = np.arange(s * 0.9, s * 1.1, 0.2 * s / 1000)
    h_values = [IapwsModel(p=point.p , s=_s).h for _s in s_values]
    return s_values, h_values

def _get_isoterm_steam(point:IapwsModel) -> Tuple[List[float], List[float]]:
    t = point.t
    p = point.p
    s = point.s
    s_max = s * 1.2
    s_min = s * 0.8
    p_values = np.arange(p * 0.8, p * 1.2, (0.4 * p) / 1000)
    h_values = np.array([(IapwsModel(p=p_ , t=t)).h for p_ in p_values])
    s_values = np.array([(IapwsModel(p=p_, t=t)).s for p_ in p_values])
    mask = (s_values >= s_min) & (s_values <= s_max)
    return s_values[mask], h_values[mask]

def _get_isoterm_two_phases(point: IapwsModel) -> Tuple[List[float], List[float]]:
    x = point.x
    p = point.p    
    x_values = np.arange(x * 0.9, min(x * 1.1, 1), (1 - x) / 1000)
    h_values = np.array([IapwsModel(p=p, x=_x).h for _x in x_values])
    s_values = np.array([IapwsModel(p=p, x=_x).s for _x in x_values])
    return s_values, h_values

def get_isoterm(point) -> Tuple[List[float], List[float]]:
     if point.phase == 'Two phases':
            return _get_isoterm_two_phases(point)
    
     return _get_isoterm_steam(point)

def plot_isolines(ax: plt.Axes, point: IapwsModel) -> None:
    s_isobar, h_isobar = get_isobar(point)
    s_isoterm, h_isoterm = get_isoterm(point)
    ax.plot(s_isobar, h_isobar, color='green', label='Изобара')
    ax.plot(s_isoterm, h_isoterm, color='blue', label='Изотерма')

def plot_points(ax: plt.Axes, points: List[IapwsModel]) -> None:
        for point in points:
            ax.scatter(point.s, point.h, s=50, color="red")
            plot_isolines(ax, point)

def get_humidity_constant_line(
        point: IapwsModel,
        max_p: float,
        min_p: float,
        x: Optional[float]=None
    ) -> Tuple[List[float], List[float]]:
        _x = x if x else point.x
        p_values = np.arange(min_p, max_p, (max_p - min_p) / 1000)
        h_values = np.array([IapwsModel(p=_p, x=_x).h for _p in p_values])
        s_values = np.array([IapwsModel(p=_p, x=_x).s for _p in p_values])
        return s_values, h_values

def plot_humidity_lines(ax: plt.Axes, points: List[IapwsModel]) -> None:
    pressures = [point.p for point in points]
    min_pressure = min(pressures) if min(pressures) > 700/1e6 else 700/1e6
    max_pressure = max(pressures) if max(pressures) < 22 else 22
    for point in points:
        if point.phase == 'Two phases':
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure, x=1)
            ax.plot(s_values, h_values, color="gray")
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure)
            ax.plot(s_values, h_values, color="gray", label='Линия сухости')
            ax.text(s_values[10], h_values[10], f'x={round(point.x, 2)}')

def plot_hs_diagram(ax: plt.Axes, points: List[IapwsModel]) -> None:
    plot_points(ax, points)
    plot_humidity_lines(ax, points)
    ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
    ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
    ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
    ax.legend()
    ax.grid()
    legend_without_duplicate_labels(ax)

def plot_h_s_diagram( _point_0, point_0, point_1t, point_1, _point_overheating, point_overheating, point_2, point_2t):
    fig, ax  = plt.subplots(1, 1, figsize=(15, 15))
    plot_hs_diagram(
        ax,
        points=[_point_0, point_0, point_1t, point_1, _point_overheating, point_overheating, point_2, point_2t]
    )
    plot_process(ax, points=[_point_0, point_0, point_1], color='black')
    plot_process(ax, points=[_point_overheating, point_overheating, point_2], color='black')
    plot_process(ax, points=[_point_0, point_0, point_1t], alpha=0.5, color='grey')
    plot_process(ax, points=[_point_overheating, point_overheating, point_2t], alpha=0.5, color='grey')
