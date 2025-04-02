#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:57:52 2025

@author: natalie
"""

import brian2 as b2
import json 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle
import os
import scipy
from scipy.linalg import circulant
from tqdm import tqdm

import pandas as pd
from IPython.display import display, Markdown

from brian2 import BrianLogger
BrianLogger.suppress_name('resolution_conflict', 'adjusted_width') # suppress warnings about variables that exist in Brian network and global namespace

from tools import  bin_matrix, despine, draw_synapses_fixed_indegree, get_LIF_rate, get_spkstats, gridline, NpEncoder, \
        plot_array, plot_raster, ProgressBar 
    

my_rcParams = json.load(open('settings/matplotlib_my_rcParams.txt'))
matplotlib.rcParams.update(my_rcParams)

Hz, ms, mV, nA = b2.Hz, b2.ms, b2.mV, b2.nA

cm = 1/2.54

#%% aux functions 


def create_EIchain_synapses_w_fixed_indegree(seed, N_features, Ne, Ni, 
                                             p_ee_in, p_ee_out, p_ei_in, p_ei_out,
                                             p_ii_in, p_ii_out, p_ie_in, p_ie_out):
    print('Creating adjacency matrix with fixed indegree...', end=' ')
    
    np.random.seed(seed) # set seed for random synapse drawing
    
    ## inputs to E
    K_ee_in = int( p_ee_in * Ne ) # inputs from exc peers 
    See_in_ipost = np.zeros((N_features, Ne * K_ee_in)) # postsyn indices 
    See_in_ipre  = np.zeros((N_features, Ne * K_ee_in))  # presyn indices 
    
    K_ee_out = int( p_ee_out * Ne*2 ) # inputs from 2 exc neighbors resp.
    See_out_ipost = np.zeros((N_features, Ne * K_ee_out)) # postsyn indices 
    See_out_ipre  = np.zeros((N_features, Ne * K_ee_out))  # presyn indices 
    
    K_ei_in = int( p_ei_in * Ni ) # inputs from inh peers 
    Sei_in_ipost = np.zeros((N_features, Ne * K_ei_in)) # postsyn indices 
    Sei_in_ipre  = np.zeros((N_features, Ne * K_ei_in))  # presyn indices 
    
    K_ei_out = int( p_ei_out * Ni*2 ) # inputs from 2 inh neighbors resp.
    Sei_out_ipost = np.zeros((N_features, Ne * K_ei_out)) # postsyn indices 
    Sei_out_ipre  = np.zeros((N_features, Ne * K_ei_out))  # presyn indices 
    
    ## inputs to I
    K_ii_in = int( p_ii_in * Ni ) # inputs from inh peers 
    Sii_in_ipost = np.zeros((N_features, Ni * K_ii_in)) # postsyn indices 
    Sii_in_ipre  = np.zeros((N_features, Ni * K_ii_in))  # presyn indices 
    
    K_ii_out = int( p_ii_out * Ni*2 ) # inputs from 2 inh neighbors resp.
    Sii_out_ipost = np.zeros((N_features, Ni * K_ii_out)) # postsyn indices 
    Sii_out_ipre  = np.zeros((N_features, Ni * K_ii_out))  # presyn indices 
    
    K_ie_in = int( p_ie_in * Ne ) # inputs from exc peers 
    Sie_in_ipost = np.zeros((N_features, Ni * K_ie_in)) # postsyn indices 
    Sie_in_ipre  = np.zeros((N_features, Ni * K_ie_in))  # presyn indices
    
    K_ie_out = int( p_ie_out * Ne*2 ) # inputs from 2 exc neighbors resp.
    Sie_out_ipost = np.zeros((N_features, Ni * K_ie_out)) # postsyn indices 
    Sie_out_ipre  = np.zeros((N_features, Ni * K_ie_out)) # presyn indices 
    
    for k in tqdm(range(N_features)):
        ### inputs to E -------------------------------------------------------------------------------------------------------------------------
        # EE-in: inputs from same exc population 
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ne, Npost = Ne, p=p_ee_in, autapses = False, return_list=False) # all synapse pairs for population k
        See_in_ipre[k]  = ipre.copy() + k*Ne # offset by ix of first neuron in population k     
        See_in_ipost[k] = ipost.copy() + k*Ne # offset by ix of first neuron in population k     
        
        # EE-out: inputs from other exc populations 
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ne*2, Npost = Ne, p=p_ee_out, return_list=False) # all synapse pairs for population k
        ix_left, ix_right = ipre.copy() < Ne, ipre.copy() >= Ne
        ipre[ix_left]  = (ipre[ix_left] + (k-1)*Ne) % (Ne*N_features) # population left of k
        ipre[ix_right] = (ipre[ix_right] + k*Ne) % (Ne*N_features) # population right of k
        See_out_ipre[k] = ipre.copy()
        See_out_ipost[k] = ipost.copy() + k*Ne # offset by ix of first neuron in population k     
        
        # EI-in: inputs from partner I population 
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ni, Npost = Ne, p=p_ei_in, return_list=False) # all synapse pairs for population k
        Sei_in_ipre[k]  = ipre.copy() + k*Ni # offset by ix of first neuron in population k     
        Sei_in_ipost[k] = ipost.copy() + k*Ne # offset by ix of first neuron in population k     
        
        # EI-out: inputs from neighbor I populations
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ni*2, Npost = Ne, p=p_ei_out, return_list=False) # all synapse pairs for population k
        ix_left, ix_right = ipre.copy() < Ni, ipre.copy() >= Ni
        ipre[ix_left]  = (ipre[ix_left] + (k-1)*Ni) % (Ni*N_features) # population left of k
        ipre[ix_right] = (ipre[ix_right] + k*Ni) % (Ni*N_features) # population right of k
        Sei_out_ipre[k]  = ipre.copy()
        Sei_out_ipost[k] = ipost.copy() + k*Ne # offset by ix of first neuron in population k     
    
        # ### inputs to I  -------------------------------------------------------------------------------------------------------------------------
        # II-in: inputs from same inh population 
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ni, Npost = Ni, p=p_ii_in, autapses = False, return_list=False) # all synapse pairs for population k
        Sii_in_ipre[k]  = ipre.copy() + k*Ni # offset by ix of first neuron in population k     
        Sii_in_ipost[k] = ipost.copy() + k*Ni # offset by ix of first neuron in population k     
        
        # II-out: inputs from other inh populations 
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ni*2, Npost = Ni, p=p_ii_out, return_list=False) # all synapse pairs for population k
        ix_left, ix_right = ipre.copy() < Ni, ipre.copy() >= Ni
        ipre[ix_left]  = (ipre[ix_left] + (k-1)*Ni) % (Ni*N_features) # population left of k
        ipre[ix_right] = (ipre[ix_right] + k*Ni) % (Ni*N_features) # population right of k
        Sii_out_ipre[k] = ipre.copy()
        Sii_out_ipost[k] = ipost.copy() + k*Ni # offset by ix of first neuron in population k     
        
        # IE-in: inputs from partner E population 
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ne, Npost = Ni, p=p_ie_in, return_list=False) # all synapse pairs for population k
        Sie_in_ipre[k]  = ipre.copy() + k*Ne # offset by ix of first neuron in population k     
        Sie_in_ipost[k] = ipost.copy() + k*Ni # offset by ix of first neuron in population k     
        
        # IE-out: inputs from neighbor E populations
        ipre, ipost = draw_synapses_fixed_indegree(Npre = Ne*2, Npost = Ni, p=p_ie_out, return_list=False) # all synapse pairs for population k
        ix_left, ix_right = ipre.copy() < Ne, ipre.copy() >= Ne
        ipre[ix_left]  = (ipre[ix_left] + (k-1)*Ne) % (Ne*N_features) # population left of k
        ipre[ix_right] = (ipre[ix_right] + k*Ne) % (Ne*N_features) # population right of k
        Sie_out_ipre[k]  = ipre.copy()
        Sie_out_ipost[k] = ipost.copy() + k*Ni # offset by ix of first neuron in population k     

    return (list(See_in_ipre.flatten().astype(int)), list(See_in_ipost.flatten().astype(int))), (list(See_out_ipre.flatten().astype(int)), list(See_out_ipost.flatten().astype(int))), \
           (list(Sei_in_ipre.flatten().astype(int)), list(Sei_in_ipost.flatten().astype(int))), (list(Sei_out_ipre.flatten().astype(int)), list(Sei_out_ipost.flatten().astype(int))), \
           (list(Sii_in_ipre.flatten().astype(int)), list(Sii_in_ipost.flatten().astype(int))), (list(Sii_out_ipre.flatten().astype(int)), list(Sii_out_ipost.flatten().astype(int))), \
           (list(Sie_in_ipre.flatten().astype(int)), list(Sie_in_ipost.flatten().astype(int))), (list(Sie_out_ipre.flatten().astype(int)), list(Sie_out_ipost.flatten().astype(int))) 
    
def dist_circ(i,j,N):
    ''' circular distance between feature population indices '''
    dist = i-j 
    if not np.isscalar(dist):
        if dist.squeeze().ndim == 2:
            return np.min(np.abs(dist[None,:,:] + np.arange(-1,2)[:,None,None]*N), axis=0) # simultaneously calc distances on grid i,j
        else:
            return np.min(np.abs(dist[None,:] + np.arange(-1,2)[:,None]*N), axis=0) # i or j are fixed
    else:
        return np.min(np.abs(dist + np.arange(-1,2)*N))

def f_receptive_field(ix, gamma, r0, r_baseline, N_features, ix_on):
    return r0*gamma**dist_circ(ix, ix_on, N_features) + r_baseline

def f_exp_decay(x, tau, xinfty, x0):
    return (x0 - xinfty)*np.exp(-x/tau) + xinfty

def f_thr_lin(x, x0, m):
    return m*(x-x0)*np.heaviside(x-x0,0)

def extract_weights(ix_pre, ix_post, w, N_features, N, n):
    ''' 
    ix_pre: presynaptic indices of all synaptic connections 
    ix_post: postsynaptic indices of all synaptic connections 
    w: weights of all synaptic connections 
    N_features: # features populations 
    N: scalar or tuple (different for pre and post): original number of neurons per pre/postsyn. feature population 
    n: scalar or tuple (different for pre and post): REDUCED  number of neurons TO KEEP per pre/postsyn. feature population 
    '''
    if np.isscalar(N):
        Npre = Npost = N
        npre = npost = n
    else:   
        Npre, Npost = N[0], N[1]
        npre, npost = n[0], n[1]
    
    # show weight matrix for only 1/10 of the population neurons 
    ix_show_pre = (np.arange(npre)[None,:] + Npre*np.arange(N_features)[:,None]).flatten()
    ix_show_post = (np.arange(npost)[None,:] + Npost*np.arange(N_features)[:,None]).flatten()
    mask = np.isin(ix_pre, ix_show_pre) & np.isin(ix_post, ix_show_post) # pre and post neuron must be in chosen group
    
    # map remaining indices to continuous range: 
    def map_ix(ix, N, n):
        pop_ix = ix // N # which pop does neuron belong to 
        return ix - (N-n)*pop_ix
    
    W = np.zeros((N_features * npost, N_features * npre))
    W[map_ix(ix_post[mask], Npost, npost), map_ix(ix_pre[mask], Npre, npre)] = w[mask]
    
    return W

def scale_weight(J, p, N):
    if p*N:
        w = J / (p*N) # scale normalized weight by indegree 
    else:
        w = np.nan 
    return w    

def fit_transfer_fct(r_max, params):
    I = np.arange(0, 10*params['Vthr'], .1)
    if params['input_mode'] == 'mean':
        f = get_LIF_rate(I, params['sigma_bg'], params['Vreset'], params['Vthr'], params['Vrest'], params['tm'], params['tref'])
    elif params['input_mode'] == 'gaussian_white_noise':
        sigma =  np.sqrt(params['w_ex']/2*I)  # noise corresponding to mean provided by poisson input with weight w_ex
        f = get_LIF_rate(I, sigma, params['Vreset'], params['Vthr'], params['Vrest'], params['tm'], params['tref'])
    
    # thr-lin fit only for rates up to r_max:
    I = I[f<=r_max]
    f = f[f<=r_max]
    
    mu0, g = scipy.optimize.curve_fit(f_thr_lin, I, f)[0] # fit threshold-linear curve 
    
    # plot 
    fig, ax = plt.subplots(figsize=(4,2)) 
    ax.plot(I, f, 'k')
    ax.plot(I, f_thr_lin(I, mu0, g), '--', color= 'gray')
    ax.set_xlabel('Iext (mV)')
    ax.set_ylabel('f (Hz)')
    return mu0, g    

#%% parameter description

def load_parameters(param_file, display_description=False):
    params_descriptions = \
    {'I_ex_off': '(mV) FF drive to exc neurons in unstimulated feature populations',
     'I_ex_on': '(mV) FF drive to exc neurons in stimulated feature population',
     'I_ix_off': '(mV) FF drive to inh neurons in unstimulated feature populations',
     'I_ix_on': '(mV) FF drive to inh neurons in stimulated feature population',
     'J_ee_out' : '(mV) normalized synaptic weight between excitatory neurons of different feature populations',
     'J_ee_in' : '(mV) normalized synaptic weight between excitatory neurons of the same feature population',
     'J_ei_out' : '(mV) normalized synaptic weight (<0) from inhibitory to excitatory neurons of different feature populations',
     'J_ei_in' : '(mV) normalized synaptic weight (<0) from inhibitory to excitatory neurons of the same feature population',
     'J_ie_out' : '(mV) normalized synaptic weight from excitatory to inhibitory neurons of different feature populations',
     'J_ie_in' : '(mV) normalized synaptic weight from excitatory to inhibitory neurons of the same feature population',
     'J_ii_out' : '(mV) normalized synaptic weight (<0) between inhibitory neurons of different feature populations',
     'J_ii_in' : '(mV) normalized synaptic weight (<0) between inhibitory neurons of the same feature population',
     'J_ex' : '(mV) normalized synaptic weight from input to excitatory feature population neurons',
     'J_ix' : '(mV) normalized synaptic weight from input to inhibitory feature population neurons',
     'N_features': '(int) number of feature populations',
     'Ne': '(int) number of excitatory neurons per feature population',
     'Ni': '(int) number of inhibitory neurons per feature population',
     'Nx': '(int) number of neurons per input population',
     'T': '(ms) total simulation time (Tinit + Tsim)',
     'Tinit': '(ms) simulation time for initialization (with only background input to all populations)',
     'Tsim': '(ms) simulation time with stimulation to one feature population',
     'Vreset': '(mV) reset potential',
     'Vrest': '(mV) resting membrane potential',
     'Vthr': '(mV) spike threshold',
     'circulant_ee_connectivity': '(bool) whether or not to connect the first and last feature population with synapses (periodic boundary condition)',
     'delay_e_in': '(ms) delay of excitatory synapses within the same feature population',
     'delay_e_out': '(ms) delay of excitatory synapses across different feature populations',
     'delay_i_in': '(ms) delay of inhibitory synapses within the same feature population',
     'delay_i_out': '(ms) delay of inhibitory synapses across different feature populations',
     'random_delays': 'dict of bools, indicating which syn. delays are randomly, uniformly distributed',
     'dt': '(ms) simulation time step',
     'fix_indegree': '(bool) whether or not to fix the indegree of recurrent synapses between feature population neurons',
     'input_mode': '(str) input mode. mean: only FF mean input changes. gaussian_white_noise: FF mean and variance change poisson-like. poisson: FF input is sampled from Poisson-spiking input populations.',
     'ix_on': '(int) index of feature population that receives FF stimulation',
     'nRF_target': '(int) RF size for which J_ee and I_ex_on were tuned',
     'p_ee_out': 'connection probability E-to-E across different feature populations',
     'p_ee_in': 'connection probability E-to-E within feature populations',
     'p_ei_out': 'connection probability I-to-E across different feature populations',
     'p_ei_in': 'connection probability I-to-E within feature populations',
     'p_ie_out': 'connection probability E-to-I  across different feature populations',
     'p_ie_in': 'connection probability E-to-I within feature populations',
     'p_ii_out': 'connection probability I-to-I across different feature populations',
     'p_ii_in': 'connection probability I-to-I within feature populations',
     'p_ex': 'FF connection probability between input and excitatory feature populations',
     'p_ix': 'FF connection probability between input and inhibitory feature populations',
     'seed': 'random seed set before creating and simulating network',
     'seed_adjacency': 'random seed for synaptic adjacency with fixed indegree',
     'sigma_bg': '(mV) strength of iid Gaussian white noise',
     'tm': '(ms) membrane time constant',
     'tref': '(ms) absolute refractory period',
     'r_max_target': '(Hz) peak RF rate for which J_ee and I_ex_on were tuned',
     'verbose': '(bool) enables progress bars and other output'
    }
    params = json.load(open(f'parameters/{param_file}.json')) # load parameter file
    
    if display_description:
        df = pd.concat((pd.DataFrame.from_dict(params, orient='index', columns=['value']),               
                        pd.DataFrame.from_dict(params_descriptions, orient='index', columns=['description'])), axis=1)
        display(Markdown(df.to_markdown()))
    return params

#%% network

params_default = json.load(open(f'parameters/main_E.json')) # load default parameters

class Network(): 
    ''' object for simulation results'''
    def __init__(self, params_default_file = 'main_E_nRF-11', params_adjust={}, path = None): 
        if path: # TO DO: MAYBE DELETE?
            # load existing simulation from path:
            params = json.load(open(path + 'parameters.json'))
            # assign all parameter values as attributes
            for key in params:
                setattr(self, key, params[key])  
                
            # load result arrays 
            results = np.load(path + 'results.npz', 'r')
            for key in results:
                setattr(self, key, results[key])  
            
            # load spike trains 
            with open(path + 'spike_trains_exc.pkl', 'rb') as fp:
                self.spike_trains_exc = pickle.load(fp)
            if self.Ni:
                with open(path + 'spike_trains_inh.pkl', 'rb') as fp:
                    self.spike_trains_inh = pickle.load(fp)
            
            self.path = path # remember path
        
        else: # new network instance
            self.architecture = params_default_file
            params_default = json.load(open(f'parameters/{params_default_file}.json')) # load default parameters
            params = {**params_default, **params_adjust} # replace default parameters that should be changed
            # assign all parameter values as attributes
            for key in params:
                setattr(self, key, params[key])  
                
        # derive weights for given network size:
        self.scale_weights()

        # stimulate central population unless specified otherwise
        if 'ix_on' not in vars(self).keys():
            self.ix_on = self.N_features // 2 # stimulate central population unless specified otherwise
    
        return
    
    def add_check_of_inst_rate(self): 
        # add rate monitor for explosion detection 
        G_test_rate = b2.NeuronGroup(2, 'dx/dt = -x/(tau_rate_filt*ms) : 1', method='exact', name='G_test_rate')    
        M_test_rate = b2.StateMonitor(G_test_rate, 'x', record=True, name='M_test_rate')
        S_test_rate = b2.Synapses(self.network['E'], G_test_rate, 'w = 1/Ne : 1', on_pre='x_post += w', name='S_test_rate')
        self.ix_edge = (self.ix_on - self.N_features//2 - 1) % self.N_features # index of population the furthest away from stimulation 
        ix_neurons_on   = self.ix_on*self.Ne + np.arange(self.Ne)
        ix_neurons_edge = self.ix_edge*self.Ne + np.arange(self.Ne)
        S_test_rate.connect(i=ix_neurons_on, j=0) # first neuron records spikes from all neurons of "on" population 
        S_test_rate.connect(i=ix_neurons_edge, j=1) # second neuron records spikes from all neurons of "edge" population     
        self.network.add(G_test_rate, S_test_rate, M_test_rate)    
        return
        
    def add_check_of_mean_poisson_input(self):
        G_test_Iext = b2.NeuronGroup(1, 'dv/dt = -v/(tm*ms) : volt', name='G_test_Iext', method= 'exact')
        M_test_Iext = b2.StateMonitor(G_test_Iext, 'v', record=True, name='M_test_Iext')
        S_test_Iext = b2.Synapses(self.network['Pe'], G_test_Iext, 'w: volt', on_pre='v_post += w', name='S_test_Iext')
        ix_on = self.ix_on
        ix_inputs = (np.arange(self.Nx*self.p_ex) + self.ix_on*self.Nx).astype(int) # the first Nx*p_ex cells in "on" input population
        S_test_Iext.connect(i = list(ix_inputs), j=0) # sample exactly Nx*pex inputs from "on" input population 
        S_test_Iext.w = self.w_ex*mV
        
        if not S_test_Iext.N_incoming_post.item() == self.Nx*self.p_ex:
            raise ValueError('connection error in Poisson test unit!')
        
        self.network.add(G_test_Iext, M_test_Iext, S_test_Iext)
        return    
    
    def bin_rates(self, rate_binsize = None):
        ''' bin rates into larger time bins '''
        self.dt_bin = rate_binsize

        if rate_binsize:
            rate_binsize_steps = int(rate_binsize / self.dt) # number of simulation steps to be summarized into one bin 
            nbins = int(len(self.t) / rate_binsize_steps) # number of bins 
            self.t_binned = np.arange(nbins) * rate_binsize            
            self.rates_binned = bin_matrix(self.rates_raw, (nbins, self.N_features), mode='mean')
            if self.Ni:
                self.rates_binned_inh = bin_matrix(self.rates_raw_inh, (nbins, self.N_features), mode='mean')
        else:
            self.t_binned, self.rates_binned, self.rates_binned_inh = [], [], []
        
        return  
    def check_stability(self, explosion_cutoff_rate, edge_cutoff_rate):
        mean_rate_center, mean_rate_edge = (self.network['M_test_rate'].x / (self.tau_rate_filt*ms) / Hz)[:,-1] # filtered est of inst rate        
        current_time = (self.network['M_test_rate'].t/ms)[-1]
        
        if mean_rate_center > explosion_cutoff_rate: # check for high firing rates
            print(f"{self.nRF_target=}, t={current_time}ms: Rate of stimulated population exceeds limit: {mean_rate_center:.0f} > {explosion_cutoff_rate=:.0f} Hz. ")
            self.explosion = True
        elif mean_rate_edge > edge_cutoff_rate: # check for high firing rates
            print(f"{self.nRF_target=}, t={current_time}ms: Rate of stimulated population exceeds limit: {mean_rate_edge:.0f} > {edge_cutoff_rate=:.0f} Hz. ")
            self.explosion = True
        else:
            self.explosion = False
            print(f'{self.nRF_target=}: {mean_rate_center=:.0f}, {mean_rate_edge=:.0f} Hz')
        return
    
    def create_network(self): 
        
        if self.fix_indegree:
            See_in_ix, See_out_ix, Sei_in_ix, Sei_out_ix, Sii_in_ix, Sii_out_ix, Sie_in_ix, Sie_out_ix \
                = self.load_adjacency_w_fixed_indegree() # load (or construct) connectivity for fixed indegree
        
        if self.verbose:
            print('Constructing network...')
            
        # setup Brian network
        b2.start_scope()
        # set simulation seed:
        b2.seed = self.seed
        b2.devices.device.seed(self.seed)
        np.random.seed(self.seed)
        
        # neurons
        eqs = '''
        dv/dt = ( Vrest*mV - v + Iext ) / (tm*ms) + sqrt(2/(tm*ms))*sigma_v*xi + sqrt(2/(tm*ms))*sigma_bg*mV*xi_bg : volt (unless refractory)
        Iext            : volt     # FF mean input
        sigma_v         : volt     # FF Gaussian white noise
        '''
     
        # --- excitatory populations ---------------------------------------------------------------------------
        E = b2.NeuronGroup(self.N_features * self.Ne, eqs, threshold = 'v>=Vthr*mV', reset = 'v=Vreset*mV', 
                           refractory=self.tref*ms, method='euler', name='E')
        if self.sigma_bg:
            E.v = np.clip(np.random.normal(self.Vrest, self.sigma_bg, size = E.N), -np.inf, self.Vthr)*mV
        else:
            E.v = np.random.uniform(self.Vrest, self.Vthr, size = E.N)*mV
        
        # monitors
        mon_spk = b2.SpikeMonitor(E, record=True, name='spike_monitor_exc')
        mon_pop = [None]*self.N_features
        for i in range(self.N_features):
            mon_pop[i] = b2.PopulationRateMonitor(E[i*self.Ne : (i+1)*self.Ne], name=f'rate_E{i}')

        self.network = b2.Network(E, mon_spk, *mon_pop)
        
        # --- inhibitory populations ---------------------------------------------------------------------------
        if self.Ni:
            I = b2.NeuronGroup(self.N_features * self.Ni, eqs, threshold = 'v>=Vthr*mV', reset = 'v=Vreset*mV', 
                               refractory=self.tref*ms, method='euler', name='I')
            if self.sigma_bg:
                I.v = np.clip(np.random.normal(self.Vrest, self.sigma_bg, size = I.N), -np.inf, self.Vthr)*mV
            else:
                I.v = np.random.uniform(self.Vrest, self.Vthr, size = I.N)*mV
            
            mon_spk_inh = b2.SpikeMonitor(I, record=True, name='spike_monitor_inh')
            mon_pop_inh = [None]*self.N_features
            for i in range(self.N_features):
                mon_pop_inh[i] = b2.PopulationRateMonitor(I[i*self.Ni : (i+1)*self.Ni], name=f'rate_I{i}')

            self.network.add(I, mon_spk_inh, *mon_pop_inh)
        
        # --- SYNAPSES ------------------------------------------------------------------------------------------------------------
        # get some variables into namespace that will be used in Brian eq strings below:
        Ne, Ni, N_features = self.Ne, self.Ni, self.N_features # get neuron numbers into namespace (for Brian string notation below)
        delay_e_out, delay_e_in, delay_i_out, delay_i_in = self.delay_e_out, self.delay_e_in, self.delay_i_out, self.delay_i_in
          
        # synapses from E to E: ------------------------------------------------------------------------------------------------------------       
        See = b2.Synapses(E, E, 'w: volt', on_pre = 'v_post += w', name='See') # simple delta pulses, no delay     
        if self.p_ee_in: # connections within the same feature population    
            condition = '(i!=j) and (i//Ne) == (j//Ne)' # no autapses               
            if self.fix_indegree:
                See.connect(i = See_in_ix[0], j = See_in_ix[1]) 
            else:         
                See.connect(condition, p = self.p_ee_in) # connect ER-random
            See.w[condition] = self.w_ee_in*mV # set weights
            if self.random_delays['ee']:
                See.delay[condition] = 'delay_e_in*rand() * ms' # uniformly distr. syn delays 
            else:
                See.delay[condition] = delay_e_in*ms # fixed syn delays 
        if self.p_ee_out: # connections across different feature populations
            condition = self.which_cross_population_connections(source='e', target='e')
            if self.fix_indegree:
                See.connect(i = See_out_ix[0], j = See_out_ix[1]) 
            else:
                See.connect(condition, p = self.p_ee_out) # connect ER-random
            See.w[condition] = self.w_ee_out*mV # set weights
            if self.random_delays['ee']:
                See.delay[condition] = 'delay_e_out*rand() * ms' # uniformly distr. syn delays 
            else:
                See.delay[condition] = delay_e_out*ms # fixed syn delays 
            # print(np.unique(np.array(See.delay/ms)))
        self.network.add(See)
            
        if self.Ni:
        # synapses from E to I: ------------------------------------------------------------------------------------------------------------       
            if self.p_ie_in or self.p_ie_out:    
                Sie = b2.Synapses(E, I, 'w: volt', on_pre = 'v_post += w', name='Sie')
                if self.p_ie_in: # connections within the same feature population
                    condition = '(i//Ne) == (j//Ni)'
                    if self.fix_indegree:
                        Sie.connect(i = Sie_in_ix[0], j = Sie_in_ix[1])
                    else:
                        Sie.connect(condition, p = self.p_ie_in) 
                    Sie.w[condition] = self.w_ie_in*mV 
                    if self.random_delays['ie']:
                        Sie.delay[condition] = 'delay_e_in*rand() * ms' # uniformly distr. syn delays 
                    else:
                        Sie.delay[condition] = delay_e_in*ms # fixed syn delays 
                if self.p_ie_out: # connections across different feature populations
                    condition = self.which_cross_population_connections(source='e', target='i')
                    if self.fix_indegree:
                        Sie.connect(i = Sie_out_ix[0], j = Sie_out_ix[1])
                    else:
                        Sie.connect(condition, p = self.p_ie_out) 
                    Sie.w[condition] = self.w_ie_out*mV 
                    if self.random_delays['ie']:
                        Sie.delay[condition] = 'delay_e_out*rand() * ms' # uniformly distr. syn delays 
                    else:
                        Sie.delay[condition] = delay_e_out*ms # fixed syn delays 
                # print(np.unique(np.array(Sie.delay/ms)))
                self.network.add(Sie)
                
        # synapses from I to E: ------------------------------------------------------------------------------------------------------------       
            if self.p_ei_in or self.p_ei_out:
                Sei = b2.Synapses(I, E, 'w: volt', on_pre = 'v_post += w',  name='Sei')
                if self.p_ei_in: # connections within the same feature population
                    condition = '(i//Ni) == (j//Ne)'
                    if self.fix_indegree:
                        Sei.connect(i = Sei_in_ix[0], j = Sei_in_ix[1])
                    else:
                        Sei.connect(condition, p = self.p_ei_in) 
                    Sei.w[condition] = self.w_ei_in*mV 
                    if self.random_delays['ei']:
                        Sei.delay[condition] = 'delay_i_in*rand() * ms' # uniformly distr. syn delays 
                    else:
                        Sei.delay[condition] = delay_i_in*ms # fixed syn delays 
                if self.p_ei_out: # connections across different feature populations
                    condition = self.which_cross_population_connections(source='i', target='e')
                    if self.fix_indegree:
                        Sei.connect(i = Sei_out_ix[0], j = Sei_out_ix[1])
                    else:
                        Sei.connect(condition, p = self.p_ei_out) 
                    Sei.w[condition] = self.w_ei_out*mV 
                    if self.random_delays['ei']:
                        Sei.delay[condition] = 'delay_i_out*rand() * ms' # uniformly distr. syn delays 
                    else:
                        Sei.delay[condition] = delay_i_out*ms # fixed syn delays
                # print(np.unique(np.array(Sei.delay/ms)))
                self.network.add(Sei)
                
        # synapses from I to I : ------------------------------------------------------------------------------------------------------------       
            if self.p_ii_in or self.p_ii_out:
                Sii = b2.Synapses(I, I, 'w: volt', on_pre = 'v_post += w', name='Sii')
                if self.p_ii_in: # connections within the same feature population
                    condition = '(i//Ni == j//Ni) and (i != j)' # no autapses
                    if self.fix_indegree:
                        Sii.connect(i = Sii_in_ix[0], j = Sii_in_ix[1])
                    else:
                        Sii.connect(condition, p=self.p_ii_in) # II coupling only within populations, not across
                    Sii.w[condition] = self.w_ii_in*mV
                    if self.random_delays['ii']:
                        Sii.delay[condition] = 'delay_i_in*rand() * ms' # uniformly distr. syn delays 
                    else:
                        Sii.delay[condition] = delay_i_in*ms # fixed syn delays
                if self.p_ii_out:  # connections across different feature populations
                    condition = self.which_cross_population_connections(source='i', target='i')
                    if self.fix_indegree:
                        Sii.connect(i = Sii_out_ix[0], j = Sii_out_ix[1])
                    else:
                        Sii.connect(condition, p = self.p_ii_out)
                    Sii.w[condition] = self.w_ii_out*mV 
                    if self.random_delays['ii']:
                        Sii.delay[condition] = 'delay_i_out*rand() * ms' # uniformly distr. syn delays 
                    else:
                        Sii.delay[condition] = delay_i_out*ms # fixed syn delays
                # print(np.unique(np.array(Sii.delay/ms)))
                self.network.add(Sii)    
             
        return
    
    def determine_indegree(self):
        self.Kee, self.Kei, self.Kie, self.Kii, self.Kex, self.Kix = (0,0), (0,0), (0,0), (0,0), (np.nan, np.nan), (np.nan, np.nan)
        
        if self.p_ee_out or self.p_ee_in:
            self.Kee = (np.mean(self.network['See'].N_incoming_post), np.std(self.network['See'].N_incoming_post))
        if self.Ni:
            if self.p_ie_out or self.p_ie_in:
                self.Kie = (np.mean(self.network['Sie'].N_incoming_post), np.std(self.network['Sie'].N_incoming_post))
            if self.p_ei_out or self.p_ei_in:
                self.Kei = (np.mean(self.network['Sei'].N_incoming_post), np.std(self.network['Sei'].N_incoming_post))
            if self.p_ii_out or self.p_ii_in:
                self.Kii = (np.mean(self.network['Sii'].N_incoming_post), np.std(self.network['Sii'].N_incoming_post))
        
        if self.input_mode == 'poisson':
            try:
                self.Kex = (np.mean(self.network['Sep'].N_incoming_post), np.std(self.network['Sep'].N_incoming_post))
                if self.Ni:
                    self.Kix = (np.mean(self.network['Sip'].N_incoming_post), np.std(self.network['Sip'].N_incoming_post))
            except:
                print('FF indegree only available after simulation (setup of Poisson input).')
        if self.verbose:
            print(f'Indegrees (mean, SD): {self.Kee=}, {self.Kei=}, {self.Kie=}, {self.Kii=}.')
            if self.input_mode == 'poisson':
                print(f'FF Indegrees (mean, SD): {self.Kex=}, {self.Kix=}.')
        return
    
    def determine_response_time(self, tmax = None): 
        ''' Calculate L1 loss of population rates compared to steady-state, and fit exponential decay time constant '''
        if not tmax:
            tmax = self.Tsim # time after stim onset to consider for fit of loss decay
        if self.Ne >= 2000: # default for publication 
            self.loss_L1 = np.sum(np.abs(self.rates_raw - self.rates_stat), axis=1)
            self.loss_L1_t = self.t 
            dt_loss = self.dt
        else: # use binned rates for small populations to reduce noise
            self.loss_L1 = np.sum(np.abs(self.rates_binned - self.rates_stat), axis=1)
            self.loss_L1_t = self.t_binned 
            dt_loss = self.dt_bin 
            
        self.loss_L1_init = np.mean(self.loss_L1[int(self.Tinit/2/dt_loss):int(self.Tinit/dt_loss)]) # average over 2nd half of initialization period
        start = int(self.Tinit/dt_loss)
        end   = int((self.Tinit + tmax)/dt_loss) # time window for exp fit
        t = np.arange(0, tmax, dt_loss)
        
        def exp_decay_fit(x, tau, xinfty):
            return f_exp_decay(x, tau, xinfty, self.loss_L1_init)
        
        popt = scipy.optimize.curve_fit(exp_decay_fit, t, self.loss_L1[start:end])[0] # fit loss function from stim onset onwards
        if popt[1] > self.loss_L1_init:
            self.loss_L1_min, self.tau_resp = np.nan, np.nan # loss GROWS instead of decaying (typically happens when rates fluctuate strongly)
        else:
            self.tau_resp, self.loss_L1_min = popt
        return 

    
    def extract_spike_trains(self, recorded_units_per_population = np.nan):
        '''
        store spike trains in dictionary without brian units 
        only keep few neurons per feature population (but using their original index!)
        '''
        if np.isnan(recorded_units_per_population):
            recorded_units_per_population = self.Ne # store all units 
        if self.verbose:
            print(f'Extracting spike trains, keeping {recorded_units_per_population} neurons per feature population...')
     
        self.spike_trains_exc = {}
        spike_trains_exc_brian = self.network['spike_monitor_exc'].spike_trains()
        for f in tqdm(range(self.N_features)):
            for i in range(recorded_units_per_population):
                ixe = f*self.Ne + i # index of neuron i in population f
                self.spike_trains_exc[ixe] = spike_trains_exc_brian[ixe] / b2.ms               
        
        if self.Ni:
            self.spike_trains_inh = {}
            spike_trains_inh_brian = self.network['spike_monitor_inh'].spike_trains()
            for f in tqdm(range(self.N_features)):
                for i in range(recorded_units_per_population):
                    ixi = f*self.Ni + i
                    self.spike_trains_inh[ixi] = spike_trains_inh_brian[ixi] / b2.ms 
        
        return
    def fit_RF_size(self, plot=False):
        if self.explosion:
            self.n_RF, self.gamma_fit, self.r0_fit, self.r_bl_fit = np.nan, np.nan, np.nan, np.nan
            return self.n_RF, None, None
        
        # exponential fit of stationary rates:
        def f_fit(ix, gamma, r0, r_baseline):
            return f_receptive_field(ix, gamma, r0, r_baseline, self.N_features, self.ix_on)
        
        try:
            self.gamma_fit, self.r0_fit, self.r_bl_fit \
                = scipy.optimize.curve_fit(f_fit, np.arange(self.N_features), self.rates_stat, 
                                           p0 = [.5, np.max(self.rates_stat), 0],
                                           bounds = ([0, 0, 0], [1, np.inf, np.inf]) )[0] # fit powerlaw to popualtion rates 
            
            fitting_error = np.sqrt(np.mean((f_fit(np.arange(self.N_features), self.gamma_fit, self.r0_fit, self.r_bl_fit) - self.rates_stat)**2)) / self.r0_fit
        except:
            print('RF fitting failed!')
            self.gamma_fit, self.r0_fit, self.r_bl_fit = np.nan, np.nan, np.nan
            fitting_error = np.inf 
            
        # infer RF field size: 
        d = -1/np.log(self.gamma_fit)
        self.n_RF = 2*d + 1 

        if plot:    
            fig, ax = plt.subplots()
            ax.plot(self.rates_stat, 'ko', markersize=5)
            if fitting_error < 1e100:
                ax.plot(f_fit(np.arange(self.N_features), self.gamma_fit, self.r0_fit, self.r_bl_fit), 'r.:')
            ax.set_xlabel('Feature Population')
            ax.set_ylabel('stat. rate (Hz)')
        else:
            fig = None
        if self.verbose:
            print(f'Simulated RF size: {self.n_RF:.2f}')
        return self.n_RF, fitting_error, fig
    
    def fit_transfer_fct(self, r_max):
        I = np.arange(0, 10*self.Vthr, .1)
        if self.input_mode == 'mean':
            f = get_LIF_rate(I, self.sigma_bg, self.Vreset, self.Vthr, self.Vrest, self.tm, self.tref)
        elif self.input_mode == 'gaussian_white_noise':
            sigma =  np.sqrt(self.w_ex/2*I)  # noise corresponding to mean provided by poisson input with weight w_ex
            f = get_LIF_rate(I, sigma, self.Vreset, self.Vthr, self.Vrest, self.tm, self.tref)
        
        # thr-lin fit only for rates up to r_max:
        I = I[f<=r_max]
        f = f[f<=r_max]
        
        mu0, g = scipy.optimize.curve_fit(f_thr_lin, I, f)[0] # fit threshold-linear curve 
        
        # plot 
        fig, ax = plt.subplots(figsize=(4,2)) 
        ax.plot(I, f, 'k')
        ax.plot(I, f_thr_lin(I, mu0, g), '--', color= 'gray')
        ax.set_xlabel('Iext (mV)')
        ax.set_ylabel('f (Hz)')
        return mu0, g        

    def get_average_poisson_input(self):
        I_ex, I_ix = np.zeros(self.N_features), np.zeros(self.N_features)
        for i in range(self.N_features):
            I_ex[i] = self.tm/1000*self.J_ex*np.mean((self.network[f'rate_PE{i}'].rate[self.network[f'rate_PE{i}'].t/ms > self.Tinit])/Hz)
            if self.Ni:
                I_ix[i] = self.tm/1000*self.J_ix*np.mean((self.network[f'rate_PI{i}'].rate[self.network[f'rate_PI{i}'].t/ms > self.Tinit])/Hz)
        return I_ex, I_ix
    def get_path_to_adjacency(self):
        ''' generate a hash depending on network parameters and return full path to where pregenerated adjacency matrices are stored 
        '''
        # one long filename unique for given network sizes and connection probabilities
        path_to_adjacency = f'settings/adjacency_fixed_indegree/seed-{self.seed_adjacency}_Nf-{self.N_features}_Ne-{self.Ne}_Ni-{self.Ni}_' + \
                            f'p-{self.p_ee_in*100:.0f}-{self.p_ee_out*100:.0f}-{self.p_ei_in*100:.0f}-{self.p_ei_out*100:.0f}-{self.p_ii_in*100:.0f}-{self.p_ii_out*100:.0f}-{self.p_ie_in*100:.0f}-{self.p_ie_out*100:.0f}.npz'
        return path_to_adjacency 
    
    def get_population_rate_arrays(self, sigma_smooth = 2):
        ''' all population rates in one array '''
        nsteps = len(self.t)

        # exc rates:
        self.rates_raw = np.zeros((nsteps , self.N_features))
        self.rates = np.zeros((nsteps, self.N_features))
        for i in range(self.N_features):
            self.rates_raw[:,i] = self.network[f'rate_E{i}'].rate/Hz
            self.rates[:,i] = self.network[f'rate_E{i}'].smooth_rate(window='gaussian', width=sigma_smooth*ms)/Hz
        
        # inh rates 
        if self.Ni and self.network['rate_I0'].active:
            self.rates_raw_inh = np.zeros((nsteps , self.N_features))
            self.rates_inh = np.zeros((nsteps , self.N_features))
            for i in range(self.N_features):
                self.rates_raw_inh[:,i] = self.network[f'rate_I{i}'].rate/Hz
                self.rates_inh[:,i] = self.network[f'rate_I{i}'].smooth_rate(window='gaussian', width=sigma_smooth*ms)/Hz
            
        return
    
    def get_stationary_feature_population_rates(self, offset = 500):
        ''' average population rates from time Tinit + offset onwards '''
        if not self.explosion:
            if offset >= 0:
                averaging_interval = self.t >= self.Tinit + offset
            elif offset < 0: # counting from end of simulaiton
                averaging_interval = self.t > self.t.max() - offset
            # exc rates:
            self.rates_stat = np.mean(self.rates_raw[averaging_interval], axis=0) # stationary rate
            if self.Ni and self.network['rate_I0'].active:
                self.rates_stat_inh = np.mean(self.rates_raw_inh[averaging_interval], axis=0) # stationary rates inh
        else:
            self.rates_stat = [] 
            if self.Ni:
                self.rates_stat_inh = []
        return            
              

    

    
    def load_adjacency_w_fixed_indegree(self):
        '''
        load or create an adjacency matrix with fixed indegree 
        returns:
            for each pathway (e.g. ee_in: E-to-E connections WITHIN populations):
                a tuple of lists with PRE and POSTsynaptic indices for all synapses
        '''
        if not self.circulant_ee_connectivity:
            raise ValueError('This option has not been implemented yet in create_EIchain_synapses_w_fixed_indegree() !!!')                
        path_to_adjacency = self.get_path_to_adjacency()
        
        if os.path.exists(path_to_adjacency): # a matching adjacency matrix has already been generated and stored 
            if self.verbose:
                print('Load stored adjacency matrix with fixed indegree.')
            with np.load(path_to_adjacency, 'r') as data:
                See_in_ix = data['See_in_ix']
                See_out_ix=data['See_out_ix'] 
                Sei_in_ix=data['Sei_in_ix'] 
                Sei_out_ix=data['Sei_out_ix'] 
                Sii_in_ix=data['Sii_in_ix'] 
                Sii_out_ix=data['Sii_out_ix'] 
                Sie_in_ix=data['Sie_in_ix'] 
                Sie_out_ix=data['Sie_out_ix']
        else: # adjacency matrix mustbe newly generated and stored
            See_in_ix, See_out_ix, Sei_in_ix, Sei_out_ix, Sii_in_ix, Sii_out_ix, Sie_in_ix, Sie_out_ix \
                = create_EIchain_synapses_w_fixed_indegree(self.seed_adjacency, self.N_features, self.Ne, self.Ni, 
                                                           p_ee_in = self.p_ee_in, p_ee_out = self.p_ee_out, 
                                                           p_ei_in = self.p_ei_in, p_ei_out = self.p_ei_out,  
                                                           p_ii_in = self.p_ii_in, p_ii_out = self.p_ii_out,
                                                           p_ie_in = self.p_ie_in, p_ie_out = self.p_ie_out)
            if not os.path.exists('settings/adjacency_fixed_indegree/'):
                os.makedirs('settings/adjacency_fixed_indegree/') # create folder for storing adjacency matrices
            np.savez(path_to_adjacency, See_in_ix = See_in_ix, See_out_ix=See_out_ix, Sei_in_ix=Sei_in_ix, Sei_out_ix=Sei_out_ix, Sii_in_ix=Sii_in_ix, 
                     Sii_out_ix=Sii_out_ix, Sie_in_ix=Sie_in_ix, Sie_out_ix=Sie_out_ix, seed=self.seed_adjacency)
            print('--> stored in settings/adjacency_fixed_indegree/')
        return See_in_ix, See_out_ix, Sei_in_ix, Sei_out_ix, Sii_in_ix, Sii_out_ix, Sie_in_ix, Sie_out_ix
    
    def plot_connectivity(self, ne=None, ni=None, mark_feature_populations = True):
        ''' extract and plot weight matrix reduced to only ne / ni neurons per exc/inh feature population (save memory) '''
        if not ne:
            ne = self.Ne
        if not ni:
            ni = self.Ni
        
        Wee = extract_weights(ix_pre = np.array(self.network['See'].i), ix_post = np.array(self.network['See'].j), 
                              w = np.array(self.network['See'].w/mV), N_features = self.N_features, N = self.Ne, n = ne) 
        
        if self.Ni:
            Wei = extract_weights(ix_pre = np.array(self.network['Sei'].i), ix_post = np.array(self.network['Sei'].j), 
                                  w = np.array(self.network['Sei'].w/mV), N_features = self.N_features, 
                                  N = (self.Ni, self.Ne), n = (ni, ne)) 
            Wie = extract_weights(ix_pre = np.array(self.network['Sie'].i), ix_post = np.array(self.network['Sie'].j), 
                                  w = np.array(self.network['Sie'].w/mV), N_features = self.N_features, 
                                  N = (self.Ne, self.Ni), n = (ne, ni))
            if self.p_ii_in or self.p_ii_out:
                Wii = extract_weights(ix_pre = np.array(self.network['Sii'].i), ix_post = np.array(self.network['Sii'].j), 
                                      w = np.array(self.network['Sii'].w/mV), N_features = self.N_features, 
                                      N = self.Ni, n = ni) 
            else:
                Wii = np.zeros((ni*self.N_features, ni*self.N_features))
            
            W = np.block([[Wee, Wei], [Wie, Wii]])
        else:
            W = Wee
            
        # plot
        wmax = np.max(W) #np.max(np.abs(W))
        wmin = -.1 if (W>=0).all() else np.min(W)
        fig, ax = plt.subplots(1,2,figsize=(3,2), gridspec_kw={'width_ratios': [20,1]}) 
        im = ax[0].imshow(W,  interpolation=None, cmap = plt.cm.seismic, norm=matplotlib.colors.TwoSlopeNorm(vcenter=0, vmax=wmax, vmin=wmin))
        cbar = fig.colorbar(im, cax= ax[1], label='weight [mV]')
        if len(np.unique(W)) < 10:
            cbar.ax.set_yticks(list(np.unique(W)))  
        ax[0].set_xlabel('presynaptic neuron')
        ax[0].set_ylabel('postsynaptic neuron')
        # mark feature populations: 
        if mark_feature_populations:
            for x in np.arange(0, ne*self.N_features, ne):
                ax[0].axvline(x, 0, 1, color='lightgray', lw=.5)
                ax[0].axhline(x, 0, 1, color='lightgray', lw=.5)
            if self.Ni:
                for x in ne*self.N_features + np.arange(0, ni*self.N_features, ni):
                    ax[0].axvline(x, 0, 1, color='lightgray', lw=.5)
                    ax[0].axhline(x, 0, 1, color='lightgray', lw=.5)

        return W, fig, ax

    
    def plot_fI_curve(self, plot=True, I_mV = []):
        if not len(I_mV):
            # range of inputs from 0 to 2x largest mean input:             
            I_mV = np.arange(0, 2*np.max(self.network['E'].Iext/mV), .01)
        # largest sigma:
        sigma = np.sqrt(np.max(self.network['E'].sigma_v/mV)**2 + self.sigma_bg**2)
        f = get_LIF_rate(I_mV, sigma, self.Vreset, self.Vthr, self.Vrest, self.tm, self.tref)
        if plot:    
            plt.figure()
            plt.plot(I_mV, f, 'k', label=f'{sigma=}mV')
        if len(np.unique(self.network['E'].sigma_v/mV)) > 1 : 
            # smallest sigma
            sigma = np.sqrt(np.min(self.network['E'].sigma_v/mV)**2 + self.sigma_bg**2)
            f = [f, get_LIF_rate(I_mV, sigma, self.Vreset, self.Vthr, self.Vrest, self.tm, self.tref)]
            if plot:    
                plt.figure()            
                plt.plot(I_mV, f[1], 'r--', label=f'{sigma=}mV')
                plt.legend()
        return I_mV, f
    
    def plot_sim(self, color_by_distance = True, plot_spks = True, Ne_show_spikes=10, Ni_show_spikes=10, 
                 rate_visualization = 'binned'): 
        ''' plot simulation '''
        
        if color_by_distance:
            # feature colors dep on distance from focus population
            dist_from_stim = dist_circ(np.arange(self.N_features), self.ix_on, self.N_features)
            self.feature_colors = matplotlib.cm.nipy_spectral(dist_from_stim / dist_from_stim.max())
                
        # plot -----------------------------------------------------------------------------------------------------    
        fig, axes = plt.subplots(3 + 2*(self.Ni>0), 2, figsize=(7,4), sharex='col',
                                 gridspec_kw = {'wspace': .05, 'width_ratios':[5, 'n_RF' in vars(self).keys()]})
        despine(axes)
        for i in range(axes.shape[0]):
           if not i==2:
               axes[i,1].remove()             
        
        # --- L1 loss
        ax = axes[:,0]
        ax[0].set_ylabel('L1 loss (Hz)') # shown here for raw rates!
        if not self.explosion:
            ax[0].plot(self.loss_L1_t, self.loss_L1, 'k')
            if 'tau_resp' in vars(self).keys():
                ax[0].plot([self.Tinit/2, self.Tinit], np.ones(2)*self.loss_L1_init, 'r')
                ax[0].plot(self.loss_L1_t[self.loss_L1_t>=self.Tinit], 
                           f_exp_decay(self.loss_L1_t-self.Tinit, self.tau_resp, self.loss_L1_min, self.loss_L1_init)[self.loss_L1_t>=self.Tinit],
                           'r', label=r'$\tau_\mathrm{resp}$: '+f'{self.tau_resp:.1f}ms')
            ax[0].legend(loc='upper right', borderaxespad=0, borderpad=0)
        
        # --- exc rates:
        ax[1].set_ylabel('E rates (Hz)')
        if rate_visualization == 'smoothed':
            plot_array(self.t, self.rates.T, ax[1], colors = self.feature_colors)
        elif rate_visualization == 'binned': 
            plot_array(self.t_binned, self.rates_binned.T, ax[1], colors = self.feature_colors) 
        else:
            plot_array(self.t, self.rates_raw.T, ax[1], colors = self.feature_colors)

        
        # --- exc spikes:
        if self.network['spike_monitor_exc'].active and plot_spks:
            if Ne_show_spikes < self.Ne: # only show spikes of some example neurons 
                hide_units = (self.Ne*np.arange(self.N_features)[:,None] + np.arange(Ne_show_spikes, self.Ne+1)[None,:]).flatten()
                f_idx = lambda x: x % self.Ne + Ne_show_spikes*(x // self.Ne)
                ylim = [-.5, self.N_features*Ne_show_spikes+.5]
            else: # show spikes of all neurons
                hide_units = []
                f_idx = lambda x: x 
                ylim = [-.5, self.N_features*self.Ne+.5]
            
            ax[2].set_ylabel('E neuron')
            plot_raster(self.spike_trains_exc, ax[2], s=1.5, unit=1, f_color = lambda x: self.feature_colors[x//self.Ne],
                        f_idx = f_idx,
                        hide_units= list(hide_units))
            ax[2].set_ylim(ylim)
            plt.setp(ax[2].get_xticklabels(), visible=True)
        
        # --- RF 
        if not self.explosion and (not np.isnan(self.n_RF)):
            ### stat rates histogram
            axes[2,1].barh(np.arange(self.N_features), self.rates_stat, height=.9, color = 'gray')
            axes[2,1].plot(f_receptive_field(np.arange(self.N_features), self.gamma_fit, self.r0_fit, self.r_bl_fit, self.N_features, self.ix_on),
                           np.arange(self.N_features), 'k:')
            axes[2,1].set_ylim([-.5, self.N_features-.5])
            axes[2,1].set_xlabel('stat. rate [Hz]')
            axes[2,1].text(1,1,r'$n_\mathrm{RF}^\mathrm{fit}$ : ' + f'{self.n_RF:.1f}', va = 'top', ha='right', transform = axes[2,1].transAxes)
            plt.setp(axes[2,1].get_yticklabels(), visible=False)
            plt.setp(axes[2,1].get_xticklabels(), visible=True)
        else:
            axes[2,1].remove()
        
        # --- inh rates and spikes
        if self.Ni:
            ax[3].set_ylabel('I rates (Hz)')
            if rate_visualization == 'smoothed':
                plot_array(self.t, self.rates_inh.T, ax[3], colors = self.feature_colors)
            elif rate_visualization == 'binned': 
                plot_array(self.t_binned, self.rates_binned_inh.T, ax[3], colors = self.feature_colors) 
            else:
                plot_array(self.t, self.rates_raw_inh.T, ax[3], colors = self.feature_colors)
            
            if self.network['spike_monitor_inh'].active and plot_spks:
                if Ni_show_spikes < self.Ni:
                    hide_units = (self.Ni*np.arange(self.N_features)[:,None] + np.arange(Ni_show_spikes, self.Ni+1)[None,:]).flatten()
                    f_idx = lambda x: x % self.Ni + Ni_show_spikes*(x // self.Ni)
                    ylim = [-.5, self.N_features*Ni_show_spikes+.5]
                else:
                    hide_units = []
                    f_idx = lambda x: x 
                    ylim = [-.5, self.N_features*self.Ni+.5]
                    
                ax[4].set_ylabel('I neuron')
                plot_raster(self.spike_trains_inh, ax[4], s=1.5, unit=1, f_color = lambda x: self.feature_colors[x//self.Ni],
                            f_idx = f_idx,
                            hide_units= list(hide_units))
                ax[4].set_ylim(ylim)            

        ax[-1].set_xlabel('time (ms)')
        return fig, axes

    

    def plot_stimulation(self):
        fig, ax = plt.subplots(2,2,figsize=(6,2), sharex='col', sharey='row')
        despine(ax)
        ax[0,0].set_ylabel('mean drive (mV)')
        ax[1,0].set_ylabel('SD (mV)')
        if self.input_mode == 'poisson':
            I_ex, I_ix = self.get_average_poisson_input()
            ax[0,0].plot(I_ex, 'k.-')
            gridline([self.sigma_bg], ax[1,0], lw=1, label=r'$\sigma_\mathrm{bg}$') 
            ax[1,0].plot(np.sqrt(self.w_ex/2*I_ex) , 'k.-', label=r'$\sigma_\mathrm{poi}$')
            ax[1,0].legend()
            ax[-1,0].set_xlabel('Exc feature population')
            if self.Ni:
                ax[0,1].plot(I_ix, 'k.-')
                ax[1,1].plot(np.sqrt(self.w_ix/2*I_ix) , 'k.-' )
                gridline([self.sigma_bg], ax[1,1], lw=1, label=r'$\sigma_\mathrm{bg}$') 
                ax[-1,1].set_xlabel('Inh feature population')
            else:
                ax[:,1].remove()
        else:
            ax[0,0].plot(self.network['E'].Iext/mV, 'k.-')
            gridline([self.sigma_bg], ax[1,0], lw=1, label=r'$\sigma_\mathrm{bg}$') 
            ax[1,0].plot(self.network['E'].sigma_v/mV, 'k.-', label=r'$\sigma_\mathrm{v}$')
            ax[1,0].legend()
            ax[-1,0].set_xlabel('Exc feature neuron')
            if self.Ni:
                ax[0,1].plot(self.network['I'].Iext/mV, 'k.-')
                ax[1,1].plot(self.network['I'].sigma_v/mV, 'k.-')
                gridline([self.sigma_bg], ax[1,1], lw=1) 
                ax[-1,1].set_xlabel('Inh feature neuron')
            else:
                ax[0,1].remove()
                ax[1,1].remove()
        ax[0,0].set_ylim(bottom=0)
        return fig, ax 
    
    def postproc(self, sigma_smooth = 2, offset=500, recorded_units_per_population = 10, rate_binsize = 1):
        if self.verbose:
            print('Postprocessing: ')

        self.extract_spike_trains(recorded_units_per_population = recorded_units_per_population)        
        self.get_population_rate_arrays(sigma_smooth = sigma_smooth)
        self.bin_rates(rate_binsize = rate_binsize)
        self.get_stationary_feature_population_rates(offset = offset) # find stationary rates of features populations 
        self.fit_RF_size() # fit RF size
        
        if not self.explosion: 
            if self.network['spike_monitor_exc'].active:
                self.single_neuron_statistics(offset=offset) # compute single neuron firing rates, CV, ..
        
            self.determine_response_time() 
           
        return
    
    def return_parameters(self, params_to_be_added = ['Tinit', 'Tsim', 'T', 'record_from', 
                'Kei', 'Kee', 'Kie', 'Kii', 'Kex', 'Kix']):
        params = {}
        for key in params_default.keys():
            try:
                params[key] = vars(self)[key]
            except:
                print('No default param ' + key)
        # add input info if available:
        for key in params_to_be_added:
            try:
                params[key] = vars(self)[key]
            except:
                pass
        return params
    
    def sanity_checks_postsim(self):
        self.determine_indegree() # check indegrees (incl FF connections for POisson input)
        
        if self.input_mode == 'poisson': # confirm mean input level was correct for Poisson input
            # mean input, directly sampled by test neuron:
            Iext_sample = np.mean(self.network['M_test_Iext'].v[0][self.network['M_test_Iext'].t/b2.ms > self.Tinit + 50]/b2.mV) # average voltage of stimulated sample neuron after Tinit  
            print(f'{Iext_sample=}')
            if np.abs(Iext_sample / self.I_ex_on - 1 ) > 0.05:
                raise Warning('Sampled I_ex deviates more than 5% from target I_ex_on! Is the input wiring correct?')

            # mean input determined from measured Poisson firing rate: 
            I_ex, I_ix = self.get_average_poisson_input()
            # checks:
            if not np.isclose(I_ex[self.ix_on], self.I_ex_on, atol=0.1):
                print(f'{np.isclose(I_ex[self.ix_on], self.I_ex_on, atol=0.1)=}')
            if not np.isclose(I_ex[self.ix_on-1], self.I_ex_off, atol=0.1):
                print(f'{np.isclose(I_ex[self.ix_on-1], self.I_ex_off, atol=0.1)=}')
            if self.Ni:
                if not np.isclose(I_ix[self.ix_on], self.I_ix_on, atol=0.1):
                    print(f'{np.isclose(I_ix[self.ix_on], self.I_ix_on, atol=0.1)=}')
                if not np.isclose(I_ix[self.ix_on-1], self.I_ix_off, atol=0.1):
                    print(f'{np.isclose(I_ix[self.ix_on-1], self.I_ix_off, atol=0.1)=}')
        return
    
    def scale_weights(self):
        ''' set weights depending on indegree as w = J / (p*N) '''
        for source in ['e', 'i']: # excitatory, inhibitory source population 
            for target in ['e', 'i']: # excitatory or inhibitory target population 
                for pathway in ['in', 'out']: # within or across feature populations 
                    # derive weights for given indegrees: 
                    J = vars(self)[f'J_{target}{source}_{pathway}'] # normalized weight, independent of indegree 
                    p = vars(self)[f'p_{target}{source}_{pathway}'] # connection probability for the given pathway 
                    N = vars(self)[f'N{source}'] # number of neurons in presyn population
                    w = scale_weight(J, p, N)
                    setattr(self, f'w_{target}{source}_{pathway}', w) # set the weight as parameter
                    
        source = 'x' # external source population 
        for target in ['e', 'i']: # excitatory or inhibitory target population 
            # derive weights for given indegrees: 
            J = vars(self)[f'J_{target}{source}'] # normalized weight, independent of indegree 
            p = vars(self)[f'p_{target}{source}'] # connection probability for the given pathway 
            N = vars(self)[f'N{source}'] # number of neurons in presyn population
            w = scale_weight(J, p, N)
            setattr(self, f'w_{target}{source}', w) # set the weight as parameter
        return

    def set_poisson_input_rates(self):
        ''' set rates of poisson input populations to produce correct mean input I_ex_on/off'''
        # rates of a single poisson input unit (depending on postsyn feature population and stimulation state)
        rate_ex_on_per_unit  = self.I_ex_on / (self.tm/1000*self.J_ex) 
        rate_ex_off_per_unit = self.I_ex_off / (self.tm/1000*self.J_ex)
        
        if rate_ex_on_per_unit > 1000 / self.dt: 
            raise ValueError(f'With {self.Nx*self.p_ex} presynaptic Poisson neurons per neuron, each Poisson neuron must still fire at {rate_ex_on_per_unit=}Hz, which is beyond 1/dt. Increase the size Nx of Poisson input pools!')
        
        # Poisson rates of input pools to EXC feature populations:
        rates_ex_poisson_init = np.ones(self.N_features)*rate_ex_off_per_unit # poisson rates (per unit), depending on input pool, during init phase
        rates_ex_poisson_stim = rates_ex_poisson_init.copy() 
        rates_ex_poisson_stim[self.ix_on] = rate_ex_on_per_unit  # poi rates during stimulation phase 
        
        # setup time- and neuron-dependent array of poisson firing rates:
        self.rates_pex = b2.TimedArray(np.array([rates_ex_poisson_init, rates_ex_poisson_stim])*Hz, dt = self.Tinit*ms) # dim0: time, dim1: neuron
        
        if self.Ni: 
            # same for inhibition 
            # rates of a single poisson input unit (depending on postsyn feature population and stimulation state)
            rate_ix_on_per_unit  = self.I_ix_on / (self.tm/1000*self.J_ix) 
            rate_ix_off_per_unit = self.I_ix_off / (self.tm/1000*self.J_ix)
            
            # Poisson rates of input pools to INH feature populations:
            rates_ix_poisson_init = np.ones(self.N_features)*rate_ix_off_per_unit # poisson rates (per unit), depending on input pool, during init phase
            rates_ix_poisson_stim = rates_ix_poisson_init.copy() 
            rates_ix_poisson_stim[self.ix_on] = rate_ix_on_per_unit  # poi rates during stimulation phase 
            
            self.rates_pix = b2.TimedArray(np.array([rates_ix_poisson_init, rates_ix_poisson_stim])*Hz, dt = self.Tinit*ms) # dim0: time, dim1: neuron 
        
        return
    
    def setup_poisson_input(self):
        ''' setup and tune poisson input populations '''
        self.set_poisson_input_rates() # derive rates for Poisson input pools given Iext_on/off and FF weights 
        Nx, Ne, Ni = self.Nx, self.Ne, self.Ni # get variables into namespace
        
        # Poisson input pools for EXC feature populations:
        Pe = b2.PoissonGroup(self.Nx*self.N_features, rates = 'rates_pex(t, i//Nx)', name='Pe') # rate depends on time (</> Tinit) and postsyn target feature population 
        Sep = b2.Synapses(Pe, self.network['E'], 'w: volt', on_pre = 'v_post += w', name='Sep')
        Sep.connect(condition = '(i//Nx) == (j//Ne)', p=self.p_ex) # all feature populations draw only from their own pool of presyn poisson inputs
        Sep.w = self.w_ex*mV # same FF weight everywhere 
        
        # monitor the Poisson activity as sanity check or for plotting
        mon_poi = [None]*self.N_features 
        for f in range(self.N_features): 
            mon_poi[f] = b2.PopulationRateMonitor(Pe[f*self.Nx : (f+1)*self.Nx], name=f'rate_PE{f}')
        
        self.network.add(Pe, Sep, *mon_poi) # add Poisson pools, synapses and monitors to network 
        self.add_check_of_mean_poisson_input()
        
        if self.verbose:
            print(f'Poisson input with indegree {(np.mean(self.network["Sep"].N_incoming_post), np.std(self.network["Sep"].N_incoming_post))} and off/on rates: {self.rates_pex(0*ms,0):.2f}, {self.rates_pex(2*self.Tinit*ms,self.ix_on):.2f}Hz.')
        
        # Poisson input pools for INH feature populations:
        if self.Ni: 
            Pi = b2.PoissonGroup(self.Nx*self.N_features, rates = 'rates_pix(t, i//Nx)', name='Pi') # rate depends on time (</> Tinit) and postsyn target feature population 
            Sip = b2.Synapses(Pi, self.network['I'], 'w: volt', on_pre = 'v_post += w', name='Sip')
            Sip.connect(condition = '(i//Nx) == (j//Ni)', p=self.p_ix) # all feature populations draw only from their own pool of presyn poisson inputs
            Sip.w = self.w_ix*mV # same FF weight everywhere 
            
            # monitor the Poisson activity as sanity check or for plotting 
            mon_poi_i = [None]*self.N_features 
            for f in range(self.N_features): 
                mon_poi_i[f] = b2.PopulationRateMonitor(Pi[f*self.Nx : (f+1)*self.Nx], name=f'rate_PI{f}')
            
            self.network.add(Pi, Sip, *mon_poi_i)
        return

    
    def set_external_stimulus(self, level):
        if self.input_mode == 'poisson':
            return # everything taken care of already when setting up Poisson pools and rates 
        elif level == 'off':
            # set mean: 
            self.network['E'].Iext = self.I_ex_off * mV # background stimulation
            if self.Ni: # only background input to interneurons
                self.network['I'].Iext = self.I_ix_off * mV # background stimulation
            # set variance: 
            if self.input_mode == 'gaussian_white_noise': # add Gaussian white noise depending on mean input:
                self.network['E'].sigma_v = np.sqrt(self.w_ex/2*self.I_ex_off) * mV # all neurons from same feature population receive same noise level (iid)
                if self.Ni: 
                    self.network['I'].sigma_v = np.sqrt(self.w_ix/2*self.I_ix_off) * mV 
        elif level == 'on':
            # input per exc feature population: 
            Iext_e = np.ones(self.N_features)*self.I_ex_off 
            Iext_e[self.ix_on] = self.I_ex_on
            # input per inh feature population: 
            Iext_i = np.ones(self.N_features)*self.I_ix_off 
            Iext_i[self.ix_on] = self.I_ix_on
            self.network['E'].Iext = np.repeat(Iext_e, self.Ne) * mV # all neurons from same feature population receive same mean input 
            if self.Ni:
                self.network['I'].Iext = np.repeat(Iext_i, self.Ni) * mV # all neurons from same feature population receive same mean input 
            if self.input_mode == 'gaussian_white_noise': # add Gaussian white noise depending on mean input:
                self.network['E'].sigma_v = np.sqrt(self.w_ex/2*np.repeat(Iext_e, self.Ne)) * mV # all neurons from same feature population receive same noise level (iid)
                if self.Ni:
                    self.network['I'].sigma_v = np.sqrt(self.w_ix/2*np.repeat(Iext_i, self.Ni)) * mV
        return
                       
    def simulate(self, Tsim, Tinit = 500, record_from = [], record_vars = ('v'), n_intv=4, 
                 explosion_cutoff_rate = 300, edge_cutoff_rate = 40, tau_rate_filt=50, 
                 print_schedule = True):
        
        # store input info:
        self.Tinit = Tinit
        self.Tsim = Tsim
        self.T = Tinit + Tsim
        self.tau_rate_filt = tau_rate_filt
        
        b2.defaultclock.dt = self.dt*ms # set time step
        
        if 'network' not in vars(self).keys():
            self.create_network() # setup Brian network
    
        # record neuron variables (voltage etc)
        if len(record_from):
            self.record_from = record_from
            self.mon_state = b2.StateMonitor(self.network['E'], record_vars, record = record_from, name = 'state_monitor')
            self.network.add(self.mon_state) 
        
        # setup Poisson inputs if necessary         
        if self.input_mode == 'poisson':
            self.setup_poisson_input()

        # control for rate explosions:
        self.add_check_of_inst_rate() # add test neurons that measure filtered inst. rate of focus and edge population to detect rate explosions

        # simulate    
        if print_schedule and self.verbose:
            print(self.network.scheduling_summary()) # print schedule of Brian simulation 
        if self.verbose:
            print('Equilibrate...')        
        self.set_external_stimulus('off')
        self.network.run(Tinit*ms, namespace=vars(self),  report=ProgressBar() if self.verbose else None) # run for Tinit to let network equilibrate
        self.check_stability(explosion_cutoff_rate = explosion_cutoff_rate, edge_cutoff_rate = edge_cutoff_rate)

        if not self.explosion:
            if self.verbose:
                print('Stimulus on...')
            self.set_external_stimulus('on')
            
            # run for Tsim (split into n_intv intervals) with stimulation on
            l_intv =  Tsim / n_intv # length of one sim interval
            for intv in range(n_intv): # split simulation time into n_intv intervals 
                self.network.run(l_intv*ms, namespace=vars(self),  report=ProgressBar() if self.verbose else None ) # simulate
                self.check_stability(explosion_cutoff_rate = explosion_cutoff_rate, edge_cutoff_rate = edge_cutoff_rate)
                if self.explosion:
                    break

        self.t = self.network['rate_E0'].t/b2.ms # time stamp array 
        self.sanity_checks_postsim()
        return 

    def single_neuron_statistics(self, rate_estimate = 'spk_count', offset = 500):
        r_isi, _, self.CV, _, r_count = get_spkstats(self.network['spike_monitor_exc'].spike_trains(), unit = b2.ms, 
                                                     offset = self.Tinit + offset, T = self.T-self.Tinit-offset)
        if rate_estimate == 'spk_count':
            self.rates_stat_per_neuron = r_count 
        elif rate_estimate == 'inverse_isi': 
            self.rates_stat_per_neuron = r_isi 
        return
    
    def store_spike_trains(self, path):
        if self.verbose:
            print('Storing spike trains...')
        
        with open(path + 'spike_trains_exc.pkl', 'wb') as fp:
            pickle.dump(self.spike_trains_exc, fp)         
        
        if self.Ni:
            with open(path + 'spike_trains_inh.pkl', 'wb') as fp:
                pickle.dump(self.spike_trains_inh, fp)
        
        return
    
    def store(self, path, allow_overwrite= False):
        if os.path.exists(path):
            if not allow_overwrite:
                raise ValueError(f'Path {path} already exists!')
        else: 
            os.makedirs(path)
        
        self.store_parameters(path)
        self.store_results(path)
        self.store_spike_trains(path)
        return
    
    def store_parameters(self, path, filename=''):
        params = self.return_parameters()
        out_file = open(path + filename + "parameters.json", "w")
        json.dump(params, out_file, indent = 6, cls=NpEncoder)
        out_file.close()
        return
    
    def store_results(self, path):
        results_to_store = ['rates_raw', 'rates', 'rates_stat', 'n_RF', 'gamma_fit', 'r0_fit', 'r_bl_fit', 'explosion'] 
        if self.Ni:
            results_to_store += ['rates_raw_inh', 'rates_inh', 'rates_stat_inh']
            
        results = {}
        for key in results_to_store:
            results[key] = vars(self)[key]

        np.savez(path + 'results.npz', **results)
        return 
     

        

    
    def which_cross_population_connections(self, source, target): 
        '''
        Return string condition defining across-population coupling. Used for establishing random ER synapses with Brian. 
        Brian notation: 
            i: index of PREsynaptic neuron 
            j: index of POSTsynaptic neuron 
        '''        
        if self.circulant_ee_connectivity: # connect edge populations to each other
            condition = ' or '.join([f'(abs(i//N{source} - j//N{target}) == 1)',              # target and source pop are direct neighbors 
                                     f'(i//N{source}==0 and j//N{target}==N_features-1)',     # source pop has ix 0, target pop is last
                                     f'(j//N{target} == 0 and i//N{source} == N_features-1)']) # target pop has ix 0, source pop is last
        else:
            condition = f'abs(i//N{source} - j//N{target}) == 1' # target and source pop are direct neighbors 
        return condition  
    
#%% analytical tuning of network
def tune_network_analytically(n_RF, r_max, g, mu0, tm, J_ee_in = np.nan):
    '''
    n_RF:   target RF size 
    r_max:  (Hz) target RF peak rate 
    g:      (Hz/mV) slope of single neuron fI curve
    tm:     (ms) membrane time constant 
    J_ee_in: (mV) within-population coupling strength (if it is fixed, independent of RF size)
    '''

    d = (n_RF -1 ) // 2 
    gamma = np.exp(-1/d)
    
    Iext_off = mu0 # background mean input 
    
    # synaptic coupling strength: 
    if np.isnan(J_ee_in): # default: both across AND within coupling are varied depending on nRF
        if np.isscalar(gamma):
            J_ee = 1/(tm/1000 * g * np.sum(gamma**np.r_[-1,0,1])) #rates_target[ix_on + 2] / (tm/1000*g*rates_target[ix_on+1 : ix_on+4].sum()) 
        else:
            J_ee = 1/(tm/1000 * g * np.sum(gamma[:,None]**(np.r_[-1,0,1][None,:]), axis=1))
        Iext_on = mu0 + r_max / g * (1-gamma**2)/(1+gamma+gamma**2)
    else: # within-pop coupling remains fixed
        if np.isscalar(gamma):
            J_ee = (1- tm/1000 * g * J_ee_in) / (tm/1000 * g * np.sum(gamma**np.r_[-1,1])) #rates_target[ix_on + 2] / (tm/1000*g*rates_target[ix_on+1 : ix_on+4].sum()) 
        else:
            J_ee = (1- tm/1000 * g * J_ee_in) / (tm/1000 * g * np.sum(gamma[:,None]**(np.r_[-1,1][None,:]), axis=1))    
        Iext_on = mu0 + r_max / g * (1-gamma**2)/(1+gamma**2) * (1-J_ee_in*tm/1000*g)

    return J_ee, Iext_on, Iext_off





    
 