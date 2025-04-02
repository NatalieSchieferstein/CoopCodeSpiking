#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:29:10 2024

@author: natalie
"""
import json
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import scipy 
from scipy import special
import sys 
from tqdm import tqdm

pi = np.pi

#%% MISC
def bin_matrix(A, new_shape, mode='sum'):
  ''' bin the array A along one or two axes, such that the new matrix has shape "new_shape"
  perform the binning by either summing up the elements or averaging them.
  

  Parameters
  ----------
  A : numpy matrix 1 or 2D
  new_shape : scalar (if A is 1D), tuple or list (if A is 2D)
    shape for the new, binned matrix
  mode: 'sum' or 'mean'
    TAKE THE SUM OR MEAN OF ALL ELEMENTS OF THE SAME BIN.
  Returns
  -------
  Anew: binned matrix of new_shape

  '''
  if len(A.shape) == 1:
    A = A[:,None]
    new_shape = (new_shape, 1)
  if (np.array(A.shape) % np.array(new_shape) != 0).any():
    raise ValueError('Current matrix shape must be multiple of new shape!')
    
  ar = A.reshape((new_shape[0], A.shape[0]//new_shape[0], new_shape[1], A.shape[1]//new_shape[1]))
  if mode == 'sum':
    Anew = np.sum(np.sum(ar,axis=-1), axis=1)
  elif mode == 'mean':
    Anew = np.mean(np.mean(ar,axis=-1), axis=1)
  if not Anew.shape == new_shape:
    raise ValueError('something wrong in the implementation')
  
  return Anew.squeeze()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

#%% Brian
class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")
            
def plot_raster(spktrain, ax, label=False, s=1.5, unit=1, f_color = lambda x: 'tab:blue', f_idx = lambda x: x, hide_units=[], show_progress=False):
    
    for i in tqdm(spktrain.keys()):
        if i not in hide_units:
            spktimes = spktrain[i]/unit
            neuron_ix = f_idx(i)
            ax.scatter(spktimes,  neuron_ix*np.ones(len(spktimes)), s=s, marker='.', color=f_color(i), lw=0)
    if label:
        ax.set_ylabel('unit')
        ax.set_xlabel('time [ms]')
    return ax

def get_spkstats(spktrain, unit=1, offset=0, T=np.nan):
  ''' 
  r: mean firing rate
  CV: measure for regularity of single unit firing, returns coefficient of variation of ISIs
  spktrain: spike train dictionary from Brian
  N: #neurons
  T: ms, total time window (with offset subtracted)
  '''
  N = len(spktrain.keys())
  CV = np.ones(N)*np.nan
  r_isi = np.zeros(N)
  r_count = np.zeros(N)
  for i in range(N):
    t_spk = spktrain[i]/unit # ms
    t_spk = t_spk[t_spk>=offset]
    if len(t_spk) > 1:
      r_count[i] = len(t_spk) / (T/1000)
      ISI = np.diff(t_spk)
      if ISI.size > 1:
        CV[i] = np.std(ISI)/np.mean(ISI)
        r_isi[i] = 1000/np.mean(ISI)
      elif ISI.size==1:
        r_isi[i] = 1000/np.mean(ISI)
  CVmean = np.nanmean(CV)
  r_isi_mean = np.mean(r_isi)  
  return r_isi, r_isi_mean, CV, CVmean, r_count


def draw_synapses_fixed_indegree(Npre, Npost, p, autapses=True, return_list=True):
    '''
    Npre: size of presyn population
    Npost: size of postsyn population
    p: connection probability
    autapses: allow connections from index i  to itself? (only relevant if pre and post-syn populations are identical)
    returns:
      i: list of presynaptic indices
      j: list of postsynaptic indices
      (to be used in Brians Synapse.connect(i=, j=))
      
    '''
    
    Npre = int(Npre)
    Npost = int(Npost)  
    K_in = int(p*Npre) # in-degree
    j = np.repeat(np.arange(Npost), K_in) # postsyn indices
    i = np.zeros((Npost, K_in))+np.nan
    sources_all = np.arange(Npre) # list of all source neurons
    for m in range(Npost):
      mask = np.ones(Npre).astype(bool)
      if not autapses:
        mask[m] = False
      i[m,:] = np.random.choice(sources_all[mask], K_in, replace=False)
    i = i.flatten()
    
    if return_list:
        i = list(i.astype(int))
        j = list(j.astype(int))
    return i, j

#%% PLOTTING
def despine(ax, which=['top', 'right']):
  if type(ax) in [list, np.ndarray]:
    for axi in ax:
      despine(axi, which=which)
  else:
    for spine in which:
      ax.spines[spine].set_visible(False)
      if spine == 'bottom': # also remove xticks 
        ax.tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom=False)
        
        
def gridline(coords, ax, axis='y', zorder=-10, color='gray', lw=.5, label='', linestyle=':'):
  if np.isscalar(coords):
    coords = list([coords])
  for c in coords:
    if axis=='y':
      if type(ax) in [list, np.ndarray]:
        for axi in ax:
          axi.axhline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
      else:
        ax.axhline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
    else:
      if type(ax) in [list, np.ndarray]:
        for axi in ax:
          axi.axvline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
      else:
        ax.axvline(c, lw=lw, linestyle=linestyle, color=color, zorder=zorder, label=label)
  return

def plot_array(x, y, ax, idx=[], labels=[], colors=[], linestyle = '-', marker='', legend=True, lw=1):
  ''' plot a family of graphs x-y into ax with colors and labels
  INPUT:
    x, y: dimension 0: trials, dimension 1: plotting-dimension
  '''
  if not x.shape[-1] == y.shape[-1]: 
    raise ValueError('x and y must be aligned in last (plotting) dimension!')
  # broadcast if necessary:
  if len(x.shape)<len(y.shape):
    x = np.repeat(x[None,:], y.shape[0], axis=0)
  elif len(y.shape)<len(x.shape):
    y = np.repeat(y[None,:], x.shape[0], axis=0)
  if not len(labels):
    labels = [None]*x.shape[0]
    legend = False
  if not len(colors):
    colidx = np.linspace(1,0,x.shape[0], endpoint=True)
    colors = plt.cm.viridis(colidx)
  #--- plotting ---------------------------------------------------------------
  if len(idx): #restrict all curves to a specific region
    if not idx.shape == (x.shape[0],2):
      raise ValueError('wrong indexing dimensions!')
    for x, y, c, l, ix in zip(x,y, colors, labels, idx):  
        ax.plot(x[ix[0]:ix[-1]+1],y[ix[0]:ix[-1]+1], color=c, label=l, linestyle =linestyle, marker=marker, lw=lw)
        if legend:
          ax.legend()
  else:
    for x, y, c, l in zip(x,y, colors, labels):  
        ax.plot(x,y,color=c,label=l, linestyle =linestyle, marker=marker, lw=lw)
        if legend:
          ax.legend()
  return


#%% LIF fI curve
def phi(x):
  ''' Sergi, Brunel Eq (9) bzw. Lindner Neusig Skript (5.11)'''
  return np.exp(x**2)*scipy.special.erfc(x)

def get_LIF_mfp(mu, sigma_v, vr, vthr, vrest, tm):
  '''
  general implementation of formula for constant (sigma = 0) or stochastic drive (sigma > 0)
  mean first passage time of LIF neuron
  mu: [mV]
  sigma_v: [mV] , SD of membrane potentials for diffusion without bdry conditions
  below: sigma = sqrt(2Dtm) = sqrt(2)*sigma_v
  tm: [ms]
  careful: This function does NOT add the refrac period!
  '''
  # substract resting potential from reset and threshold
  sigma = np.sqrt(2)*sigma_v
  
  if vrest:
    vr = vr- vrest
    vthr = vthr - vrest
  if not sigma: # constant drive
      if mu > vthr:
        T = tm*np.log((mu-vr)/(mu-vthr))
      else:
        T = np.infty
  else: # stochastic drive
      # take integral
      bdry_up = (mu-vr)/sigma
      bdry_low = (mu-vthr)/sigma
      # store the values for numerical check of the integration:
      if np.min((bdry_low, bdry_up)) > 10: #2.5 : # use asymptotic approximation of erfc function (phi)
        # print(mu)
        # print('using asymptotic for rate integral...')
        T = tm*(np.log((mu-vr)/(mu-vthr))+(sigma**2)/4*(1/((mu-vr)**2)-1/((mu-vthr)**2)))  # mean 1st passage time, Eq (10)
      else:
        # print('using scipy.integrate.quad...')
        T = tm*np.sqrt(pi)*scipy.integrate.quad(phi,bdry_low,bdry_up)[0] # mean 1st passage time, Eq (10)  
  return T # ms

def get_LIF_rate(mu, sigma_v, vr, vthr, vrest, tm, tref):
  '''
  general implementation of formula for stationary firing rate of LIF driven by white noise
  mu: [mV]
  sigma_v: [mV] , SD of membrane potentials for diffusion without bdry conditions
      sometimes in other notation: sigma = sqrt(2Dtm) = sqrt(2)*sigma_v
   tm, tref: [ms]
  '''
  if np.isscalar(mu):
    T = get_LIF_mfp(mu, sigma_v, vr, vthr, vrest, tm)
    r = 1000/(T+tref)
  else:
    r = np.zeros(mu.size)
    if np.isscalar(sigma_v):
        sigma_v = np.ones(mu.size)*sigma_v
    for i in range(mu.size):
      T = get_LIF_mfp(mu[i], sigma_v[i], vr, vthr, vrest, tm)
      r[i] = 1000/(T+tref)
  return r # Hz