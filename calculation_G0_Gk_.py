import pandas as pd
from scipy import interpolate
from iapws_model import IapwsModel
from iapws import IAPWS97 as gas
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy.optimize import fsolve


MPa = 10 ** 6
kPa = 10 ** 3
unit = 1 / MPa
MW = 10 ** 6

FILE_PATH = 'zzz.csv'
FILE_PATH_pr_stator_loss = 'pr_pot.csv'
FILE_PATH_sum_stator_loss = 'sum_pot.csv'
FILE_PATH_pr_rotor_loss = 'ksi_r_pr.csv'
FILE_PATH_sum_rotor_loss = 'loss_ksi_sum_rotor.csv'

to_kelvin = lambda x: x + 273.15 if x else None


def get_function_for_find_ksi(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    x = df['x'].values
    y = df[' y'].values
    f = interpolate.interp1d(x, y)
    return f
    
def get_function_for_find_loss(path):
    df = pd.read_csv(path)
    x = df['x'].values
    y = df[' y'].values
    f = interpolate.interp1d(x, y)
    return f

def get_losses_in_stator_from_plots(
    M_1t,
    b_1,
    length_nozzle    
):   
    interpolation_function_pr_stator_loss = get_function_for_find_loss(FILE_PATH_pr_stator_loss)
    interpolation_function_sum_stator_loss = get_function_for_find_loss(FILE_PATH_sum_stator_loss)
    x_coeff_sum_stator_loss = b_1 / length_nozzle
    y_coeff_sum_stator_loss = interpolation_function_sum_stator_loss(x_coeff_sum_stator_loss)
    x_coeff_pr_stator_loss = M_1t
    y_coeff_pr_stator_loss = interpolation_function_pr_stator_loss(x_coeff_pr_stator_loss)
    ksi_pr_stator = y_coeff_pr_stator_loss * 10 ** (-2)
    ksi_sum_stator = y_coeff_sum_stator_loss * 10 ** (-2)
    ksi_k_stator = ksi_sum_stator - ksi_pr_stator
    return (
        ksi_sum_stator,
        ksi_pr_stator,
        ksi_k_stator
    )
    
def get_losses_in_rotor_from_plots(
    M2t,
    b_2,
    l_2   
):   
    interpolation_function_pr_rotor_loss = get_function_for_find_loss(FILE_PATH_pr_rotor_loss)
    interpolation_function_sum_rotor_loss = get_function_for_find_loss(FILE_PATH_sum_rotor_loss)
    x_coeff_pr_rotor_loss = M2t
    y_coeff_pr_rotor_loss = interpolation_function_pr_rotor_loss(x_coeff_pr_rotor_loss)
    x_coeff_sum_rotor_loss = b_2 / l_2
    y_coeff_sum_rotor_loss = ((5.3*(b_2 / l_2))+4)
    ksi_pr_rotor = y_coeff_pr_rotor_loss * 10 ** (-2)
    ksi_sum_rotor = y_coeff_sum_rotor_loss * 10 ** (-2)
    ksi_k_rotor = ksi_sum_rotor - ksi_pr_rotor
    return (
        ksi_sum_rotor,
        ksi_pr_rotor,
        ksi_k_rotor
    )

def get_points_for_params(
    p_0,
    t_0,
    delta_p_0,
    p_overheating,
    t_overheating,
    delta_p_overheating,
    delta_p_1,
    p_k,
    p_feed_water,
    t_feed_water,
    internal_efficiency
    ):
    real_p_0 = p_0 - delta_p_0
    real_p1t = p_overheating + delta_p_overheating
    real_p_overheating = p_overheating - delta_p_1
    _point_0 = IapwsModel(p = p_0 * unit, t=to_kelvin(t_0))
    point_0 = IapwsModel(p=real_p_0 * unit, h=_point_0.h)
    _Point_0 = gas(P = p_0 * unit, T=to_kelvin(t_0))
    Point_0 = gas(P=real_p_0 * unit, h = _Point_0.h)
    point_1t = IapwsModel(p=real_p1t * unit, s=_point_0.s)
    hp_heat_drop = (_point_0.h - point_1t.h) * internal_efficiency
    h_1 = point_0.h - hp_heat_drop
    point_1 = IapwsModel(p=real_p1t * unit, h=h_1)
    _point_overheating = IapwsModel(p=p_overheating * unit, t=to_kelvin(t_overheating))
    point_overheating = IapwsModel(p=real_p_overheating * unit, h=_point_overheating.h)
    point_2t = IapwsModel(p=p_k * unit, s=_point_overheating.s)
    lp_heat_drop = (_point_overheating.h - point_2t.h) * internal_efficiency
    h_2 = point_overheating.h - lp_heat_drop
    point_2 = IapwsModel(p = p_k * unit, h = h_2)
    point_k_water = IapwsModel(p=p_k * unit, x=0)
    point_feed_water = IapwsModel(p=p_feed_water * unit, t=to_kelvin(t_feed_water))
    return (
        _point_0,
        point_0, 
        point_1t, 
        hp_heat_drop, 
        point_1, 
        _point_overheating, 
        point_overheating,
        point_2t,
        lp_heat_drop, 
        point_2, 
        point_k_water, 
        point_feed_water,
        Point_0,
        _Point_0
    )
    
        
def get_ksi_infinity(
    point_0,
    point_1t,
    point_2,
    _point_overheating,
    point_overheating,
    point_k_water,
    point_feed_water
    ):
    numenator_without = point_2.t * (_point_overheating.s - point_k_water.s)
    denumenator_without = (point_0.h - point_1t.h) + (point_overheating.h - point_k_water.h)
    without_part = 1 - (numenator_without / denumenator_without)
    numenator_infinity = point_2.t * (_point_overheating.s - point_feed_water.s)
    denumenator_infinity = (point_0.h - point_1t.h) + (point_overheating.h - point_feed_water.h)
    infinity_part = 1 - (numenator_infinity / denumenator_infinity)
    ksi_infinity = 1 - (without_part / infinity_part)
    return ksi_infinity


def get_ksi(
    point_0,
    point_1t,
    point_2,
    _point_overheating,
    point_overheating,
    point_k_water,
    point_feed_water
    )-> float:
    ksi_infinity = get_ksi_infinity(
        point_0=point_0,
        point_1t=point_1t,
        point_2=point_2,
        _point_overheating=_point_overheating,
        point_overheating=point_overheating,
        point_k_water=point_k_water,
        point_feed_water=point_feed_water
    )
    interpolation_function = get_function_for_find_ksi(FILE_PATH)
    x_coeff = (point_feed_water.t - point_2.t) / (to_kelvin(374.2) - point_2.t)
    y_coeff = interpolation_function(x_coeff)
    ksi = y_coeff * ksi_infinity
    return ksi


def get_efficiency(
    hp_heat_drop,
    lp_heat_drop,
    point_overheating,
    point_k_water,
    ksi
    ):
    eff_num = hp_heat_drop + lp_heat_drop
    eff_denum = hp_heat_drop + (point_overheating.h - point_k_water.h)
    efficiency = (eff_num / eff_denum) * (1 / (1 - ksi))
    return efficiency



def calculate_inout_mass_flow_rate(
    p_0,
    t_0,
    p_overheating,
    t_overheating,
    p_feed_water, 
    t_feed_water, 
    p_k, 
    electrical_power,
    internal_efficiency, 
    mechanical_efficiency, 
    generator_efficiency, 
    delta_p_0, 
    delta_p_overheating, 
    delta_p_1
):
    
    (
        _point_0,
        point_0, 
        point_1t, 
        hp_heat_drop, 
        point_1, 
        _point_overheating, 
        point_overheating,
        point_2t,
        lp_heat_drop, 
        point_2, 
        point_k_water, 
        point_feed_water,
        Point_0,
        _Point_0
    ) = get_points_for_params(
        p_0=p_0,
        t_0=t_0,
        delta_p_0=delta_p_0,
        p_overheating=p_overheating,
        t_overheating=t_overheating,
        delta_p_overheating=delta_p_overheating,
        delta_p_1=delta_p_1,
        p_k=p_k,
        p_feed_water=p_feed_water,
        t_feed_water=t_feed_water,
        internal_efficiency = internal_efficiency
    )
        
    ksi = get_ksi(
        point_0=point_0,
        point_1t=point_1t,
        point_2=point_2,
        _point_overheating=_point_overheating,
        point_overheating=point_overheating,
        point_k_water=point_k_water,
        point_feed_water=point_feed_water
    )
    
    efficiency = get_efficiency(
        hp_heat_drop=hp_heat_drop,
        lp_heat_drop=lp_heat_drop,
        point_overheating=point_overheating,
        point_k_water=point_k_water,
        ksi=ksi
    )
    estimated_heat_drop = efficiency * ((point_0.h - point_feed_water.h) + (point_overheating.h - point_1.h))
    inlet_mass_flow = electrical_power / (estimated_heat_drop * 1000 * mechanical_efficiency * generator_efficiency)
    condenser_mass_flow = (
    electrical_power /
    ((point_2.h - point_k_water.h) * 1000 * mechanical_efficiency * generator_efficiency) * ((1 / efficiency) - 1))
    return (
        inlet_mass_flow,
        condenser_mass_flow
    )

def get_params_for_nozzle(
    H_0,
    degree_of_reaction,
    point_0,
    inlet_mass_flow
):
    stator_heat_drop = (1 - degree_of_reaction) * H_0 
    h_1t = point_0.h - stator_heat_drop 
    point_1t_ = gas(h = h_1t, s = point_0.s)
    v_1t = point_1t_.v 
    p_1 = gas(h = h_1t, s = point_0.s).P 
    c_1t = (2 * 1000 * stator_heat_drop) ** 0.5 
    a_1t = point_1t_.w  
    M_1t = c_1t / a_1t
    mu_1_ = 0.97
    F_1 = (inlet_mass_flow[0] * v_1t) / (mu_1_ * c_1t) 
    return (
        stator_heat_drop,
        h_1t,
        v_1t,
        a_1t,
        c_1t,
        M_1t,
        F_1,
        p_1,
        point_1t_
    )

def get_params_for_inlet_triangle(
    H_0,
    degree_of_reaction,
    D_average,
    b_1,
    inlet_mass_flow,
    F_1,
    c_1t,
    v_1t,
    M_1t
):
    alpha_1_eff = 16
    n = 60
    u = np.pi * D_average * n
    denum_for_el_1 = np.pi * D_average * np.sin(np.deg2rad(alpha_1_eff))
    el_1 = F_1 / denum_for_el_1
    e_opt = 5 * (el_1) ** 0.5
    length_nozzle = el_1 / e_opt   
    mu_1 = 0.982 - 0.005 * (b_1 / length_nozzle)
    F1 = (inlet_mass_flow[0] * v_1t) / (mu_1 * c_1t)
    denum_for_el_1 = np.pi * D_average * np.sin(np.deg2rad(alpha_1_eff))
    el_1 = F_1 / denum_for_el_1
    e_opt = 5 * (el_1) ** 0.5
    length_nozzle = el_1 / e_opt
    (
        ksi_sum_stator,
        ksi_pr_stator,
        ksi_k_stator
    )= get_losses_in_stator_from_plots(
    M_1t,
    b_1,
    length_nozzle
    )
    mu_1 = 0.982 - 0.005 * (b_1 / length_nozzle)
    F1 = (inlet_mass_flow[0] * v_1t) / (mu_1 * c_1t)
    fi = 0.98 - 0.008 * (b_1 / length_nozzle)
    fi = (1 - (ksi_sum_stator))**0.5
    c_1 = c_1t * fi
    alpha_1_rad = np.arcsin((mu_1/fi) * (np.sin(np.deg2rad(alpha_1_eff))))
    sin_alpha1 = np.sin(alpha_1_rad)
    cos_alpha1 = np.cos(alpha_1_rad)
    w1 = (c_1 ** 2 + u ** 2 - 2 * c_1 * u * cos_alpha1) ** 0.5
    beta_1 = np.arccos((c_1 * cos_alpha1 - u) / w1)
    return (
        el_1,
        e_opt,
        length_nozzle,
        mu_1,
        fi,
        c_1,
        alpha_1_rad,
        w1,
        beta_1,
        cos_alpha1,
        sin_alpha1
    )
    
def get_alfa_y(
    D_average,
    e_opt,
    b_1
):
    alpha_1_eff = 16
    t_1_opt_ = 0.75
    z_1_ = (np.pi * D_average * e_opt)/(b_1 * t_1_opt_)
    x = int(z_1_)
    if x % 2 == 0:
        z_1 = x
    else:
        z_1 = int(x) + 1
    t_1_opt = (np.pi * D_average * e_opt)/(b_1 * z_1)
    if t_1_opt > 0.7 and t_1_opt < 0.85:
        return alpha_1_eff - 16 * (t_1_opt - 0.75) + 23.1
    elif t_1_opt < 0.7 and t_1_opt > 0.85:
        raise "Значение относительного шага выходит за пределы допустимых значений"

def get_beta_y(
    D_average,
    b_2,
    beta_2_eff
):
    t_2_opt_ = 0.55
    z_2_ = (np.pi * D_average)/(b_2 * t_2_opt_)
    y = int(z_2_)
    if y % 2 == 0:
        z_2 = y
    else:
        z_2 = int(y) + 1
    t_2_opt = (np.pi * D_average) / (b_2 * z_2)
    return beta_2_eff - 16.6 * (t_2_opt - 0.65) + 54.3, z_2

def get_mach_number_working_grid(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    point_0,
    inlet_mass_flow,
    c_1,
    h_1t,
    w1,
    length_nozzle,
    c_1t,
    p_1
):
    alpha_1_eff = 16
    rotor_heat_drop = H_0 * degree_of_reaction   
    stator_loss = (c_1t ** 2 / 2) - (c_1 ** 2 / 2)
    h_1 = h_1t + stator_loss / 1000
    h_2t = h_1 - rotor_heat_drop
    point_1_ = gas(P = p_1, h = h_1)
    point_2_t = gas(h = h_2t, s = point_1_.s)   
    w2t = (w1 ** 2 + 2 * rotor_heat_drop * 1000) ** 0.5
    overlapping = 0.003
    l_2 = length_nozzle + overlapping
    a2t = point_2_t.w
    M2t = w2t / a2t
    return (
        rotor_heat_drop,
        h_1,
        point_2_t,
        w2t,
        l_2,
        M2t,
        stator_loss,
        h_2t,
        point_1_
    )

def get_params_for_working_grid(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    b_2,
    point_0,
    inlet_mass_flow,
    point_2_t,
    w2t,
    l_2,
    e_opt,
    M2t
): 
    (
        ksi_sum_rotor,
        ksi_pr_rotor,
        ksi_k_rotor
    )=get_losses_in_rotor_from_plots(
    M2t,
    b_2,
    l_2    
    )   
    alpha_1_eff = 16
    n = 60
    u = np.pi * D_average * n
    mu_2 = 0.965 - 0.01* (b_2/l_2)
    F2 = (inlet_mass_flow[0] * point_2_t.v)/(mu_2 * w2t)
    sin_beta_2_eff = F2 / (e_opt *np.pi * D_average * l_2)
    beta_2_eff = np.arcsin(sin_beta_2_eff)
    psi_ = 0.96 - 0.014*(b_2/l_2)
    psi_ = (1 - (ksi_sum_rotor))**0.5
    w2 = w2t * psi_
    beta2_rad = np.arcsin((mu_2/psi_)* np.sin(beta_2_eff))
    sin_beta2 = np.sin(beta2_rad)
    cos_beta2 = np.cos(beta2_rad)
    c2 = (w2 ** 2 + u ** 2 - 2 * w2 * u * cos_beta2) ** 0.5
    alpha_2 =np.arccos((w2*cos_beta2 - u) / c2)
    delta_H_outlet_speed = (c2 ** 2) / 2
    rotor_loss = (w2t ** 2 / 2) - (w2 ** 2 / 2)   
    return (
        mu_2,
        F2,
        beta_2_eff,
        delta_H_outlet_speed,
        rotor_loss,
        beta2_rad,
        c2,
        alpha_2,
        w2,
        cos_beta2,
        sin_beta2,
        psi_
    )

def get_blade_efficiency(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    b_2,
    p_0,
    t_0,
    delta_p_0,
    p_overheating,
    t_overheating,
    delta_p_overheating,
    delta_p_1,
    p_k,
    electrical_power,
    mechanical_efficiency,
    generator_efficiency,
    p_feed_water,
    t_feed_water,
    internal_efficiency,
    is_tech = False
):
                        
    (
        _point_0,
        point_0, 
        point_1t, 
        hp_heat_drop, 
        point_1, 
        _point_overheating, 
        point_overheating,
        point_2t,
        lp_heat_drop, 
        point_2, 
        point_k_water, 
        point_feed_water,
        Point_0,
        _Point_0
    
    ) =  get_points_for_params(
        p_0=p_0,
        t_0=t_0,
        delta_p_0=delta_p_0,
        p_overheating=p_overheating,
        t_overheating=t_overheating,
        delta_p_overheating=delta_p_overheating,
        delta_p_1=delta_p_1,
        p_k=p_k,
        p_feed_water=p_feed_water,
        t_feed_water=t_feed_water,
        internal_efficiency = internal_efficiency
    ) 
    inlet_mass_flow = calculate_inout_mass_flow_rate(
        p_0 = p_0,
        t_0 = t_0,
        p_overheating = p_overheating,
        t_overheating = t_overheating,
        p_feed_water = p_feed_water, 
        t_feed_water = t_feed_water, 
        p_k = p_k, 
        electrical_power = electrical_power,
        internal_efficiency = internal_efficiency, 
        mechanical_efficiency = mechanical_efficiency, 
        generator_efficiency = generator_efficiency, 
        delta_p_0 = delta_p_0, 
        delta_p_overheating = delta_p_overheating, 
        delta_p_1 = delta_p_1
    )
    
    n = 60
    u = np.pi * D_average * n
    (
        stator_heat_drop,
        h_1t,
        v_1t,
        a_1t,
        c_1t,
        M_1t,
        F_1,
        p_1, 
        point_1t_
    )= get_params_for_nozzle(
        H_0 = H_0,
        degree_of_reaction = degree_of_reaction,
        point_0 = Point_0,
        inlet_mass_flow = inlet_mass_flow
    )
    (
        el_1,
        e_opt,
        length_nozzle,
        mu_1,
        fi,
        c_1,
        alfa_1_rad,
        w1,
        beta_1,
        cos_alpha1,
        sin_alpha1
    )=get_params_for_inlet_triangle(
        H_0 = H_0,
        degree_of_reaction = degree_of_reaction,
        D_average = D_average,
        b_1 = b_1,
        inlet_mass_flow = inlet_mass_flow,
        F_1 = F_1,
        c_1t = c_1t,
        v_1t = v_1t,
        M_1t = M_1t
    )
    (
        rotor_heat_drop,
        h_1,
        point_2_t,
        w2t,
        l_2,
        M2t,
        stator_loss,
        h_2t,
        point_1_
        
    )= get_mach_number_working_grid(
        H_0 = H_0,                                         
        D_average = D_average,
        degree_of_reaction = degree_of_reaction,
        b_1 = b_1,
        point_0 = Point_0,
        inlet_mass_flow = inlet_mass_flow,
        c_1 = c_1,
        h_1t = h_1t,
        w1 = w1,
        length_nozzle = length_nozzle,
        c_1t = c_1t,
        p_1 = p_1
        
    )
    (
        mu_2,
        F2,
        beta_2_eff,
        delta_H_outlet_speed,
        rotor_loss,
        beta2_rad,
        c2,
        alpha_2,
        w2,
        cos_beta2,
        sin_beta2,
        psi_
    )= get_params_for_working_grid(
        H_0 = H_0,
        D_average = D_average,
        degree_of_reaction = degree_of_reaction,
        b_1 = b_1,
        b_2 = b_2,
        point_0 = Point_0,
        inlet_mass_flow = inlet_mass_flow,
        point_2_t = point_2_t,
        w2t = w2t,
        l_2 = l_2,
        e_opt = e_opt,
        M2t = M2t
        
    )
    (
        el_1,
        e_opt,
        length_nozzle,
        mu_1,
        fi,
        c_1,
        alpha_1_rad,
        w1,
        beta_1,
        cos_alpha1,
        sin_alpha1      
    )= get_params_for_inlet_triangle(
        H_0 = H_0,
        degree_of_reaction = degree_of_reaction,
        D_average = D_average,
        b_1 = b_1,
        inlet_mass_flow = inlet_mass_flow,
        F_1 = F_1,
        c_1t = c_1t,
        v_1t = v_1t,
        M_1t = M_1t
        
    )
    num = H_0 * 1000 - stator_loss - rotor_loss - delta_H_outlet_speed
    denum = H_0 * 1000
    blade_efficiency = num / denum
    absolute_projection = c_1 * cos_alpha1 + c2 * np.cos(alpha_2)   
    blade_efficiency_ = (u * absolute_projection)/(H_0 * 1000)
    if is_tech:
        return (
            length_nozzle,
            el_1,
            F_1,
            num,
            M_1t,
            b_1,
            length_nozzle,
            M2t,
            b_2,
            l_2,
            c_1,
            cos_alpha1,
            sin_alpha1,
            u,
            w2,
            cos_beta2,
            sin_beta2,
            blade_efficiency,
            beta_2_eff,
            inlet_mass_flow,
            h_2t,
            rotor_loss,
            delta_H_outlet_speed,
            point_2_t,
            point_1t_,
            point_1_,
            psi_,
            fi,
            beta_1,
            c2,
            alpha_2,
            w1      
        )
    return blade_efficiency, blade_efficiency_ 
    
def plot_blade_efficiency_of_heat_drop(H_0, blade_eff, blade_eff_triangle):
    fig, ax  = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_title("График зависимости относительного лопаточного КПД от теоретического теплоперепада") 
    ax.set_xlabel("H_0, кДж/кг") 
    ax.set_ylabel("η_ол, %")     
    ax.plot(H_0, blade_eff * 100, label = "Лопаточный КПД через потери " ,color ="green")
    ax.plot(H_0, blade_eff_triangle * 100, label = " Лопаточный КПД через треугольник скоростей" ,color ="black", linestyle = ':')
    ax.grid()
    ax.legend()

def plot_blade_efficiency_of_u_cf(H_0, blade_eff, D_average):
    n = 60
    u = np.pi * D_average * n
    dummy_speed = (2 * H_0 * 1000) ** 0.5
    u_div_dummy_speed = u / dummy_speed
    fig, ax  = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_title("График зависимости относительного лопаточного КПД от u_сф") 
    ax.set_xlabel("u_cф") 
    ax.set_ylabel("η_ол, %")     
    ax.plot(u_div_dummy_speed, blade_eff * 100, label = "Относительный лопаточный КПД " ,color ="green")
    ax.grid()
    ax.legend()

def plot_triangles(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    b_2,
    p_0,
    t_0,
    delta_p_0,
    p_overheating,
    t_overheating,
    delta_p_overheating,
    delta_p_1,
    p_k,
    electrical_power,
    mechanical_efficiency,
    generator_efficiency,
    p_feed_water,
    t_feed_water,
    internal_efficiency    
):
    (  
        l_1, 
        el_1,
        F_1,
        num,
        M_1t,
        b_1,
        length_nozzle,
        M2t,
        b_2,
        l_2,
        c_1,
        cos_alpha1,
        sin_alpha1,
        u,
        w2,
        cos_beta2,
        sin_beta2,
        blade_efficiency,
        beta_2_eff,
        inlet_mass_flow,
        h_2t,
        rotor_loss,
        delta_H_outlet_speed,
        point_2_t,  
        point_1t_,
        point_1_,
        psi_,
        fi,
        beta_1,
        c2,
        alpha_2,
        w1
    
    )=get_blade_efficiency( 
        H_0 = H_0,
        D_average = D_average,
        degree_of_reaction = degree_of_reaction,
        b_1 = b_1,
        b_2 = b_2,
        p_0 = p_0,
        t_0 = t_0,
        delta_p_0 = delta_p_0,
        p_overheating = p_overheating,
        t_overheating = t_overheating,
        delta_p_overheating = delta_p_overheating,
        delta_p_1 = delta_p_1,
        p_k = p_k,
        electrical_power = electrical_power,
        mechanical_efficiency = mechanical_efficiency,
        generator_efficiency = generator_efficiency,
        p_feed_water = p_feed_water,
        t_feed_water = t_feed_water,
        internal_efficiency = internal_efficiency,
        is_tech = True 
    )
    c1_plot = [[0, -c_1 * cos_alpha1], [0, -c_1 * sin_alpha1]]
    u1_plot = [[-c_1 * cos_alpha1, -c_1 * cos_alpha1 + u], [-c_1 * sin_alpha1, -c_1 * sin_alpha1]]
    w1_plot = [[0, -c_1 * cos_alpha1 + u], [0, -c_1 * sin_alpha1]]
    w2_plot = [[0, w2 * cos_beta2], [0, -w2 * sin_beta2]]
    u2_plot = [[w2 * cos_beta2, w2 * cos_beta2 - u], [-w2 * sin_beta2, -w2 * sin_beta2]]
    c2_plot = [[0, w2 * cos_beta2 - u], [0, -w2 * sin_beta2]]
    fig, ax  = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(c1_plot[0], c1_plot[1], label='C_1', c='red')
    ax.plot(u1_plot[0], u1_plot[1], label='u_1', c='blue')
    ax.plot(w1_plot[0], w1_plot[1], label='W_1', c='green')
    ax.plot(w2_plot[0], w2_plot[1], label='W_2', c='green')
    ax.plot(u2_plot[0], u2_plot[1], label='u_2', c='blue')
    ax.plot(c2_plot[0], c2_plot[1], label='C_2', c='red')
    ax.set_title("Треугольник скоростей")
    ax.legend();

def get_bandage_leak_loss_pu(
    H_0,
    D_average,
    degree_of_reaction,
    l_2,
    F_1,
    blade_efficiency

):
    n = 60
    u = np.pi * D_average * n
    mu_a = 0.5
    delta_a = 0.0025
    mu_r = 0.75
    z = 5
    dummy_speed = (2 * H_0 * 1000) ** 0.5
    u_div_dummy_speed = u / dummy_speed
    d_per = D_average + l_2
    delta_r = 0.001 * d_per
    first = (mu_a * delta_a) ** (-2)
    second = z * (mu_r * delta_r) ** (-2)
    delta_eq = (first + second) ** (-0.5)
    one = np.pi * d_per * delta_eq * blade_efficiency 
    two = ( degree_of_reaction + (1.8 * l_2 / D_average)) ** 0.5
    bandage_leak_loss_pu = (one * two)/F_1
    return bandage_leak_loss_pu

def get_ventilation_loss_pu(
    H_0,
    D_average,
    el_1,
    l_1,
    sin_alpha1
):
    n = 60
    u = np.pi * D_average * n
    dummy_speed = (2 * H_0 * 1000) ** 0.5
    u_div_dummy_speed = u / dummy_speed 
    m = 1
    k_friction = 0.7 * 10 ** (-3)
    e = el_1 / l_1
    sin = sin_alpha1
    first = k_friction / sin
    second = (1 - e) / e
    third = u_div_dummy_speed ** 3 
    ventilation_loss_pu = first * second * third * m
    return ventilation_loss_pu

def get_segment_loss_pu(
    H_0,
    D_average,
    degree_of_reaction,
    l_2,
    F_1,
    beta_2_eff,
    blade_efficiency,
    b_2
):
    n = 60
    u = np.pi * D_average * n
    dummy_speed = (2 * H_0 * 1000) ** 0.5
    u_div_dummy_speed = u / dummy_speed       
    segments = 4
    beta_2_y, z_2 = get_beta_y(
        D_average = D_average,
        b_2 = b_2,
        beta_2_eff = beta_2_eff
    )
    blade_width= b_2 * np.sin(np.deg2rad(beta_2_y))
    first = (0.25 * blade_width * l_2) / F_1
    second = u_div_dummy_speed * blade_efficiency * segments
    segment_loss_pu = first * second
    return segment_loss_pu

def get_disk_leak_loss_pu(H_0, D_average, F_1):
    n = 60
    u = np.pi * D_average * n
    k_friction = 0.7 * 10 ** (-3)
    dummy_speed = (2 * H_0 * 1000) ** 0.5
    u_div_dummy_speed = u / dummy_speed    
    num_1 = (k_friction * D_average ** 2) * (u_div_dummy_speed**3)
    disk_leak_loss_pu = num_1 / F_1
    return disk_leak_loss_pu
    
def get_internal_params(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    b_2,
    p_0,
    t_0,
    delta_p_0,
    p_overheating,
    t_overheating,
    delta_p_overheating,
    delta_p_1,
    p_k,
    electrical_power,
    mechanical_efficiency,
    generator_efficiency,
    p_feed_water,
    t_feed_water,
    internal_efficiency
    
):
    (
    l_1,
    el_1,
    F_1,
    num,
    M1t,
    b_1,
    length_nozzle,
    M2t,
    b_2,
    l_2,
    c_1,
    cos_alpha1,
    sin_alpha1,
    u,
    w2,
    cos_beta2,
    sin_beta2,
    blade_efficiency,
    beta_2_eff,
    inlet_mass_flow,
    h_2t,
    rotor_loss,
    delta_H_outlet_speed,
    point_2_t,   
    point_1t_,
    point_1_,
    psi_,
    fi,
    beta_1,
    c2,
    alpha_2,
    w1
        
    ) = get_blade_efficiency( 
    H_0 = H_0,
    D_average = D_average,
    degree_of_reaction = degree_of_reaction,
    b_1 = b_1,
    b_2 = b_2,
    p_0 = p_0,
    t_0 = t_0,
    delta_p_0 = delta_p_0,
    p_overheating = p_overheating,
    t_overheating = t_overheating,
    delta_p_overheating = delta_p_overheating,
    delta_p_1 = delta_p_1,
    p_k = p_k,
    electrical_power = electrical_power,
    mechanical_efficiency = mechanical_efficiency,
    generator_efficiency = generator_efficiency,
    p_feed_water = p_feed_water,
    t_feed_water = t_feed_water,
    internal_efficiency = internal_efficiency,
    is_tech = True 
    )
    bandage_leak_loss_pu = get_bandage_leak_loss_pu(
    H_0 = H_0,
    D_average = D_average,
    degree_of_reaction = degree_of_reaction,
    l_2 = l_2,
    F_1 = F_1,
    blade_efficiency = blade_efficiency
    )
    disk_leak_loss_pu = get_disk_leak_loss_pu(
    H_0 = H_0,
    D_average = D_average,
    F_1 = F_1
    )
    segment_loss_pu = get_segment_loss_pu(
    H_0 = H_0,
    D_average = D_average,
    degree_of_reaction = degree_of_reaction,
    l_2 = l_2,
    F_1 = F_1,
    beta_2_eff = beta_2_eff,
    blade_efficiency = blade_efficiency,
    b_2 = b_2
    ) 
    ventilation_loss_pu = get_ventilation_loss_pu(
    H_0 = H_0,
    D_average = D_average,
    el_1 = el_1,
    l_1 = l_1,
    sin_alpha1 = sin_alpha1
    )
    beta_2_y, z_2 = get_beta_y(
        D_average = D_average,
        b_2 = b_2,
        beta_2_eff = beta_2_eff
    )
    periphery_leak_loss = H_0 * bandage_leak_loss_pu
    disk_friction_loss = H_0 * disk_leak_loss_pu 
    partial_losses_pu = segment_loss_pu + ventilation_loss_pu
    partial_losses = H_0 * partial_losses_pu
    losses =  disk_friction_loss + periphery_leak_loss + partial_losses
    internal_heat_drop = num/1000 - losses
    internal_efficiency = internal_heat_drop / (H_0)
    N_i = internal_heat_drop * internal_efficiency * inlet_mass_flow[0]
    return (
        internal_heat_drop,
        internal_efficiency,
        N_i,
        z_2,
        el_1/l_1,
        l_2,
        losses,
        h_2t,
        rotor_loss,
        delta_H_outlet_speed,
        point_2_t,
        fi,
        psi_
    )
       
def table_triangles(
    c_1,
    c_2,
    alfa_1,
    alfa_2,
    w1,
    w2,
    beta_1,
    beta_2
):
    table = pd.DataFrame({
        "Абсолютная скорость": [c_1, c_2],
        "Абсолютный угол": [alfa_1, alfa_2],
        "Относительная скорость": [w1, w2],
        "Относительный угол": [beta_1, beta_2],
        "Треугольник скоростей": ["На входе", "На выходе"],
    }).set_index(["Треугольник скоростей"])
    return table

def stress_of_rotor_blade(
    inlet_mass_flow,
    H_0,
    blade_efficiency,
    l_2,
    b_2,
    e_opt,
    D_average,
    W_min_atl,
    b_2_atl,
    z_2
):
    n=60
    u = np.pi * D_average * n
    W_min = W_min_atl * ((b_2/b_2_atl)**3)
    one = inlet_mass_flow * H_0 * 1000 * l_2 * blade_efficiency
    two = 2 * u * z_2 * W_min * e_opt
    bending_stress = (one / two) * 10 ** (-6)
    w = 2 * np.pi  * n
    density = 7800
    extension_stress = (0.5 * density * (w ** 2) * D_average * l_2) * 10 ** (-6)       
    return bending_stress, extension_stress
    
def get_reaction_degree(root_dor, veernost):
    return root_dor + (1.8 / (veernost + 1.8))

def get_u_cf(dor, alpha_1_rad, fi_):
    cos = np.cos(alpha_1_rad)
    return fi_ * cos / (2 * (1 - dor) ** 0.5)

def get_heat_drop(diameter, u_cf, n):
    first = (diameter / u_cf) ** 2
    second = (n / 50) ** 2
    return 12.3 * first * second

def get_points_for_no_regul(
    h_2t,
    rotor_loss,
    delta_H_outlet_speed,
    losses,
    point_2_t,
    p_overheating,
    internal_efficiency
):
    h_0 = h_2t + rotor_loss/1000 + delta_H_outlet_speed/1000 + losses/1000
    point_0_no_regul = gas(P=point_2_t.P, h = h_0)
    point_overheat_t = gas(P=p_overheating * unit, s=point_0_no_regul.s)
    full_heat_drop = point_0_no_regul.h - point_overheat_t.h
    actual_heat_drop = full_heat_drop * internal_efficiency
    h_overheat = h_0 - actual_heat_drop
    point_overheat= gas(P=p_overheating * unit, h=h_overheat)
    return point_overheat,full_heat_drop, point_0_no_regul

def get_veernost(
    avg_diam_1,
    blade_length_1,
    veernost_1,
    root_reaction_degree,
    n,
    point_0_no_regul,
    inlet_mass_flow,
    discharge_coefficient,
    alpha_1_rad,    
    overlapping,
    fi_
    ):
    while not np.isclose(avg_diam_1 / blade_length_1, veernost_1, rtol=0.01):
        
        veernost_1 = avg_diam_1 / blade_length_1
        avg_reaction_degree_1 = get_reaction_degree(root_reaction_degree, veernost_1)
        u_cf_1 = get_u_cf(avg_reaction_degree_1, alpha_1_rad, fi_)
        heat_drop_1 = get_heat_drop(avg_diam_1, u_cf_1, n)
        h1 = point_0_no_regul.h - heat_drop_1
        point_2 = gas(h=h1, s=point_0_no_regul.s)
        upper = inlet_mass_flow * point_2.v * u_cf_1
        lower = discharge_coefficient * np.sin(alpha_1_rad) * n * (np.pi * avg_diam_1) ** 2 * (1 - avg_reaction_degree_1) ** 0.5
        blade_length_1 = upper / lower
        blade_length_2 = blade_length_1 + overlapping
    return avg_diam_1 / blade_length_1, blade_length_2, point_2

def equation_to_solve(x, root_diameter, avg_diam_1, blade_length_2, point_overheat, point_2):
    return x ** 2 + x * root_diameter - avg_diam_1 * blade_length_2 * point_overheat.v / point_2.v

def linear_distribution(left, right, x):
    return (right - left) * x + left

def get_new_act_dr(
    output_speed_coeff_loss,
    heat_drops,
    internal_efficiency,
    full_heat_drop,
    n_stages
):
    actual_heat_drops = output_speed_coeff_loss * heat_drops
    mean_heat_drop = np.mean(actual_heat_drops)
    reheat_factor = 4.8 * 10 ** (-4) * (1 - internal_efficiency) * full_heat_drop * (n_stages - 1) / n_stages
    full_heat_drop * (1 + reheat_factor) / mean_heat_drop
    bias_1 = full_heat_drop * (1 + reheat_factor) - np.sum(actual_heat_drops)
    bias = bias_1 / n_stages
    new_actual_heat_drop = actual_heat_drops + bias
    return new_actual_heat_drop
    
def split_over_stages(
    D_average, 
    delta_diam,
    h_2t,
    rotor_loss,
    delta_H_outlet_speed,
    losses,
    point_2_t,
    p_overheating,
    internal_efficiency, 
    veernost_1,
    inlet_mass_flow,
    discharge_coefficient,
    alpha_1_rad,
    point_2,
    n_stages,
    root_reaction_degree
    
):
    n = 60
    overlapping = 0.003
    avg_diam_1 = D_average - delta_diam
    (
        point_overheat,
        full_heat_drop,
        point_0_no_regul 
        
    )= get_points_for_no_regul(
    h_2t = h_2t,
    rotor_loss = rotor_loss,
    delta_H_outlet_speed = delta_H_outlet_speed,
    losses = losses,
    point_2_t = point_2_t,
    p_overheating = p_overheating,
    internal_efficiency = internal_efficiency
    )
    veernost, blade_length_2, point_2 = get_veernost(
    avg_diam_1=avg_diam_1,
    blade_length_1 = 1,
    veernost_1 = veernost_1,
    root_reaction_degree =root_reaction_degree,
    n = n,
    point_0_no_regul = point_0_no_regul,
    inlet_mass_flow = inlet_mass_flow,
    discharge_coefficient = discharge_coefficient,
    alpha_1_rad = alpha_1_rad,    
    overlapping = overlapping,
    fi_ = discharge_coefficient
    )
    root_diameter = avg_diam_1 - blade_length_2
    blade_length_z = fsolve(equation_to_solve, 0.01, args = (
    root_diameter,
    avg_diam_1,
    blade_length_2,
    point_overheat,
    point_2
    )
    )[0]
    avg_diam_2 = root_diameter + blade_length_z
    x = np.cumsum(np.ones(n_stages) * 1 / (n_stages - 1)) - 1 / (n_stages - 1)
    diameters = linear_distribution(avg_diam_1, avg_diam_2 , x)
    blade_lengths = linear_distribution(blade_length_2, blade_length_z , x)
    veernosts = diameters / blade_lengths    
    reaction_degrees = get_reaction_degree(root_dor=root_reaction_degree, veernost=veernosts)    
    u_cf = get_u_cf(dor=reaction_degrees, alpha_1_rad = alpha_1_rad, fi_ = discharge_coefficient)    
    heat_drops = get_heat_drop(diameters, u_cf, n)   
    output_speed_coeff_loss = np.full_like(heat_drops, 0.95)
    output_speed_coeff_loss[0] = 1
    actual_heat_drops = output_speed_coeff_loss * heat_drops
    mean_heat_drop = np.mean(actual_heat_drops)
    reheat_factor = 4.8 * 10 ** (-4) * (1 - internal_efficiency) * full_heat_drop * (n_stages - 1) / n_stages
    virtual_stage = full_heat_drop * (1 + reheat_factor) / mean_heat_drop
    print(f"Полученное значение числа ступеней: {virtual_stage}, Заданное число ступеней: {n_stages}")
    bias = full_heat_drop * (1 + reheat_factor) - np.sum(actual_heat_drops)
    bias = bias / n_stages
    new_actual_heat_drop = actual_heat_drops + bias
    return (
        x,
        diameters,
        veernosts,
        blade_lengths,
        u_cf,
        reaction_degrees,
        new_actual_heat_drop
    )

def plot_distribution(values, ax_name):
    fig, ax = plt.subplots(1, 1, figsize=(15,5))
    ax.plot(range(1, len(values)+1), values,  marker='o')
    ax.set_xlabel("Номер ступени")
    ax.set_ylabel(ax_name)
    ax.grid()

def get_points_for_h_s_diagram(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    b_2,
    p_0,
    t_0,
    delta_p_0,
    p_overheating,
    t_overheating,
    delta_p_overheating,
    delta_p_1,
    p_k,
    electrical_power,
    mechanical_efficiency,
    generator_efficiency,
    p_feed_water,
    t_feed_water,
    internal_efficiency,
    losses
    
):
    (
    l_1,
    el_1,
    F_1,
    num,
    M1t,
    b_1,
    length_nozzle,
    M2t,
    b_2,
    l_2,
    c_1,
    cos_alpha1,
    sin_alpha1,
    u,
    w2,
    cos_beta2,
    sin_beta2,
    blade_efficiency,
    beta_2_eff,
    inlet_mass_flow,
    h_2t,
    rotor_loss,
    delta_H_outlet_speed,
    point_2_t,   
    point_1t_,
    point_1_,
    psi_,
    fi,
    beta_1,
    c2,
    alpha_2,
    w1
        
    ) = get_blade_efficiency( 
    H_0 = H_0,
    D_average = D_average,
    degree_of_reaction = degree_of_reaction,
    b_1 = b_1,
    b_2 = b_2,
    p_0 = p_0,
    t_0 = t_0,
    delta_p_0 = delta_p_0,
    p_overheating = p_overheating,
    t_overheating = t_overheating,
    delta_p_overheating = delta_p_overheating,
    delta_p_1 = delta_p_1,
    p_k = p_k,
    electrical_power = electrical_power,
    mechanical_efficiency = mechanical_efficiency,
    generator_efficiency = generator_efficiency,
    p_feed_water = p_feed_water,
    t_feed_water = t_feed_water,
    internal_efficiency = internal_efficiency,
    is_tech = True 
    )
    h_2 = h_2t + rotor_loss/1000 + delta_H_outlet_speed/1000 + losses/1000
    point_2_ = gas(P = point_2_t.P, h = h_2)
    return point_1_, point_1t_, point_2_t, point_2_

def legend_without_duplicate_labels(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plot_process(ax: plt.Axes, points: List[gas], **kwargs) -> None:
    ax.plot([point.s for point in points], [point.h for point in points], **kwargs)
    
def get_isobar(point: gas, left = 0.9, right = 1.1) -> Tuple[List[float], List[float]]:
    s = point.s
    s_values = np.arange(s * left, s * right, 0.2 * s / 1000)
    h_values = [gas(P=point.P, s=_s).h for _s in s_values]
    return s_values, h_values

def _get_isoterm_steam(point: gas) -> Tuple[List[float], List[float]]:
    t = point.T
    p = point.P
    s = point.s
    s_max = s * 1.2
    s_min = s * 0.8
    p_values = np.arange(p * 0.8, p * 1.2, 0.4 * p / 1000)
    h_values = np.array([gas(P=_p, T=t).h for _p in p_values])
    s_values = np.array([gas(P=_p, T=t).s for _p in p_values])
    mask = (s_values >= s_min) & (s_values <= s_max)
    return s_values[mask], h_values[mask]

def _get_isoterm_two_phases(point: gas) -> Tuple[List[float], List[float]]:
    x = point.x
    p = point.P
    x_values = np.arange(x * 0.9, min(x * 1.1, 1), (1 - x) / 1000)
    h_values = np.array([gas(P=p, x=_x).h for _x in x_values])
    s_values = np.array([gas(P=p, x=_x).s for _x in x_values])
    return s_values, h_values

def get_isoterm(point) -> Tuple[List[float], List[float]]:
    if point.phase == 'Two phases':
        return _get_isoterm_two_phases(point)
    return _get_isoterm_steam(point)

def plot_isolines(ax: plt.Axes, point: gas, left_p = None, right_p = None) -> None:
    s_isobar, h_isobar = get_isobar(point, left_p, right_p)
    s_isoterm, h_isoterm = get_isoterm(point)
    ax.plot(s_isobar, h_isobar, color='green', label='Изобара')
    ax.plot(s_isoterm, h_isoterm, color='blue', label='Изотерма')
    
def plot_points(ax: plt.Axes, points: List[gas], left_p = None, right_p = None) -> None:
    for point in points:
        ax.scatter(point.s, point.h, s=50, color="red")
        plot_isolines(ax, point,left_p, right_p)
        
def get_humidity_constant_line(
    point: gas,
    max_p: float,
    min_p: float,
    x: Optional[float]=None
) -> Tuple[List[float], List[float]]:
    _x = x if x else point.x
    p_values = np.arange(min_p, max_p, (max_p - min_p) / 1000)
    h_values = np.array([gas(P=_p, x=_x).h for _p in p_values])
    s_values = np.array([gas(P=_p, x=_x).s for _p in p_values])
    return s_values, h_values

def plot_humidity_lines(ax: plt.Axes, points: List[gas]) -> None:
    pressures = [point.P for point in points]
    min_pressure = min(pressures) if min(pressures) > 700/1e6 else 700/1e6
    max_pressure = max(pressures) if max(pressures) < 22 else 22
    for point in points:
        if point.phase == 'Two phases':
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure, x=1)
            ax.plot(s_values, h_values, color="gray")
            s_values, h_values = get_humidity_constant_line(point, max_pressure, min_pressure)
            ax.plot(s_values, h_values, color="gray", label='Линия сухости')
            ax.text(s_values[10], h_values[10], f'x={round(point.x, 2)}')

def plot_hs_diagram_expansion(ax: plt.Axes, points: List[gas], left_p = None, right_p = None) -> None:
    plot_points(ax, points, left_p, right_p)
    plot_humidity_lines(ax, points)
    ax.set_xlabel(r"S, $\frac{кДж}{кг * K}$", fontsize=14)
    ax.set_ylabel(r"h, $\frac{кДж}{кг}$", fontsize=14)
    ax.set_title("HS-диаграмма процесса расширения", fontsize=18)
    ax.legend()
    ax.grid()
    legend_without_duplicate_labels(ax)
    
def plot_h_s_diagram_expansion(_Point_0, Point_0, point_1_, point_1t_, point_2, point_2_t, left_p = 0.9, right_p = 1.1):
    fig, ax  = plt.subplots(1, 1, figsize=(15, 15))
    plot_hs_diagram_expansion(
        ax,
        points=[Point_0, _Point_0, point_1_, point_2], left_p = left_p, right_p = right_p
    )
    plot_process(ax, points=[Point_0,_Point_0, point_1_, point_2], color='black')

def table_of_loss( fi, psi_, delta_H_outlet_speed):
    
    d = {
    "Коэффициент скорости в сопловой решетке":[fi],
    "Коэффициент скорости в рабочей решетке":[psi_],
    "Потери с выходной скоростью": [delta_H_outlet_speed]
    }
    table = pd.DataFrame(d)
    return table
 
def table_triangles(
    H_0,
    D_average,
    degree_of_reaction,
    b_1,
    b_2,
    p_0,
    t_0,
    delta_p_0,
    p_overheating,
    t_overheating,
    delta_p_overheating,
    delta_p_1,
    p_k,
    electrical_power,
    mechanical_efficiency,
    generator_efficiency,
    p_feed_water,
    t_feed_water,
    internal_efficiency
    
):
    (
    l_1,
    el_1,
    F_1,
    num,
    M1t,
    b_1,
    length_nozzle,
    M2t,
    b_2,
    l_2,
    c_1,
    cos_alpha1,
    sin_alpha1,
    u,
    w2,
    cos_beta2,
    sin_beta2,
    blade_efficiency,
    beta_2_eff,
    inlet_mass_flow,
    h_2t,
    rotor_loss,
    delta_H_outlet_speed,
    point_2_t,   
    point_1t_,
    point_1_,
    psi_,
    fi,
    beta_1,
    c2,
    alpha_2,
    w1
        
    ) = get_blade_efficiency( 
    H_0 = H_0,
    D_average = D_average,
    degree_of_reaction = degree_of_reaction,
    b_1 = b_1,
    b_2 = b_2,
    p_0 = p_0,
    t_0 = t_0,
    delta_p_0 = delta_p_0,
    p_overheating = p_overheating,
    t_overheating = t_overheating,
    delta_p_overheating = delta_p_overheating,
    delta_p_1 = delta_p_1,
    p_k = p_k,
    electrical_power = electrical_power,
    mechanical_efficiency = mechanical_efficiency,
    generator_efficiency = generator_efficiency,
    p_feed_water = p_feed_water,
    t_feed_water = t_feed_water,
    internal_efficiency = internal_efficiency,
    is_tech = True 
    )
    table = pd.DataFrame({
        "Абсолютная скорость": [c_1, c2],
        "Абсолютный угол": [np.rad2deg(np.arccos(cos_alpha1)),np.rad2deg(alpha_2)],
        "Относительная скорость": [w1, w2],
        "Относительный угол": [np.rad2deg(beta_1), np.rad2deg(np.arcsin(sin_beta2))],
        "Треугольник скоростей": ["На входе", "На выходе"],
    }).set_index(["Треугольник скоростей"])
    return table

def table_of_points(
    _Point_0,
    Point_0,
    point_1_,
    point_1t_,
    point_2,
    point_2_t
):
    table = pd.DataFrame({
        "Давление P, МПа": [_Point_0.P, Point_0.P, point_1t_.P, point_1_.P, point_2_t.P, point_2.P],
        "Температура T, К": [_Point_0.T, Point_0.T, point_1t_.T, point_1_.T, point_2_t.T, point_2.T],
        "Удельный объем v, м^3/кг": [_Point_0.v, Point_0.v, point_1t_.v, point_1_.v, point_2_t.v, point_2.v],
        "Энтальпия h, кДж/кг": [_Point_0.h, Point_0.h, point_1t_.h, point_1_.h, point_2_t.h, point_2.h],
        "Энтропия s, кДж/(кг*K)":[_Point_0.s, Point_0.s, point_1t_.s, point_1_.s, point_2_t.s, point_2.s],
        "Точка": ["0'","0","1t", "1", "2t","2"],
    }).set_index(["Точка"])
    return table
  











