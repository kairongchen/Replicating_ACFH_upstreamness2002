#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to construct US upstreamness using 2002 Input-Output table.
I replicate the Stata do file in Antràs, Pol, et al. "Measuring the upstreamness of production and trade flows." AER Papers and Proceedings (2012): 412-16. 

The code is used for research and learning purposes.

Kairong Chen
Department of Economics
Indiana University Bloomington
Last edited: 09/04/2021
"""

#%% Change working directory
import os
# os.chdir('../../../../../../N/slate/krchen/code')
# os.chdir('Google Drive/TPU Project/code')

#%% load lib
import pandas as pd
import numpy as np

#%% Prepare USE table 
# The 2002 I-O data downloaded from the BEA is revised. So the value are
# slightly different with the raw data from ACFH

# # Option 1: Load raw data downloaded from the BEA
# f = "../data/construct_upstreamness/REV_NAICSUseDetail 4-24-08.txt"
# colspecs = [(0, 10), (10, 100), (100, 110), (110, 200), (200, 210)]
# df = pd.read_fwf(f, header=None, skiprows=1, skipfooter=19,
#                   colspecs=colspecs,
#                   index_cols=False,
#                   names=['Commodity', 'CommodityDescription', 'Industry', 
#                         'IndustryDescription', 'ProVal'])
# del f, colspecs

# Option 2: Load ACFH raw data
f_AFFHraw = '../data/construct_upstreamness/Section II - Upstreamness in the US/iousedetail.dta'
df = pd.read_stata(f_AFFHraw)
del f_AFFHraw

df = df.sort_values(['Commodity', 'Industry'])

df = df.rename(columns={'Commodity': 'io_input',
                        'Industry': 'io_output',
                        'CommodityDescription': 'io_input_name',
                        'IndustryDescription': 'io_output_name'})

# These are commodities that are inputs to other industries, but never outputs
df = df[~((df['io_input']=='S00300')|
          (df['io_input']=='S00401')|
          (df['io_input']=='S00402')|
          (df['io_input']=='S00900'))]
# =============================================================================
# * These four commodities are:
# * S00300 Noncomparable imports
# * S00401 Scrap
# * S00402 Used and Secondhand goods
# * S00900 Rest of the World Adjustment
# =============================================================================
df_IOUsetemp = df.copy()


#%% Generate lists of industries
df = df_IOUsetemp.copy()
   
df_input = df[['io_input', 'io_input_name']]
df_input = df_input.drop_duplicates()
df_input = df_input.rename(columns={'io_input': 'io_industry',
                        'io_input_name': 'io_industry_name'})

df_output = df[['io_output', 'io_output_name']]
df_output = df_output.drop_duplicates()
df_output = df_output.rename(columns={'io_output': 'io_industry',
                        'io_output_name': 'io_industry_name'})

# two industries (S00201 and S00202) are not listed as inputs --> To be added to get a square matrix!
df_io_newind_list = pd.merge(df_input, df_output, on=['io_industry', 'io_industry_name'], how='right')
del df_input, df_output

df_io_newind_list = df_io_newind_list[~((df_io_newind_list['io_industry'].str[0]=="F")|
                                      (df_io_newind_list['io_industry'].str[0]=="V"))]

df_io_newind_list = df_io_newind_list.sort_values('io_industry').reset_index(drop=True)
df_io_newind_list['newind'] = range(1, df_io_newind_list.shape[0]+1)

# --> 426 industries


#%% Create a square (use) matrix     
df = df_io_newind_list.rename(columns={'newind': 'newoutput',
                                       'io_industry': 'io_output',
                        'io_industry_name': 'io_output_name'}).copy()

df['key'] = 1
df_io_newind_list['key'] = 1
df = pd.merge(df, df_io_newind_list, on='key').drop(columns=['key']).rename(
    columns={'newind': 'newinput',
             'io_industry': 'io_input',
             'io_industry_name': 'io_input_name'})
# 426*426 observations

df = pd.merge(df, df_IOUsetemp, on=['io_input', 'io_input_name',
                                    'io_output', 'io_output_name'], how='left')

df = df[~((df['io_output'].str[0]=="F")|
          (df['io_input'].str[0]=="V"))]

df.loc[df['ProVal'].isnull(), 'ProVal'] = 0
# 70% of the coefficients are zeros

df_IOUseFullTemp = df.copy()

#%% Extracting Absorption
df = df_IOUsetemp.copy()
df = df[~((df['io_input'].str[0]=="V")|
          (df['io_output']=="F04000")|
          (df['io_output']=="F05000")|
          (df['io_output']=="F03000")
          )]
# drop exports and imports
# drop changes in inventories

df = df[['io_input', 'io_input_name', 'ProVal']].groupby(['io_input', 'io_input_name']).agg(sum).reset_index().rename(columns={'ProVal': 'absorption'})

df = df[df['absorption']!=0]

df = df.rename(columns={'io_input': 'io_industry',
                        'io_input_name': 'io_industry_name'})
df = df.sort_values('io_industry')

df_Absorption = df.copy()

#%% Compute Delta matrix 
df = df_IOUseFullTemp.copy()

# get absorption:
df = df.rename(columns={'io_input': 'io_industry'})
df = pd.merge(df, df_Absorption, on=['io_industry'], how='left')
df = df.drop(columns=['io_industry_name'])
df = df.rename(columns={'io_industry': 'io_input'})

df['delta'] = df['ProVal'] / df['absorption']
df.loc[df['delta'].isnull(), 'delta'] = 0

df = df[['io_input', 'io_input_name', 'newinput', 'io_output', 'io_output_name', 'newoutput', 'delta']]
df = df.sort_values(['io_input', 'io_output'])

df_Delta_matrix = df.copy()

#%% Calculation of Upstreamness
df = df_Delta_matrix.copy()

df['invcoeff'] = - df['delta']
df.loc[df['newinput']==df['newoutput'], 'invcoeff'] = 1. + df.loc[df['newinput']==df['newoutput'], 'invcoeff']

df = df[['newinput', 'newoutput', 'invcoeff']].sort_values(['newinput', 'newoutput'])

df_IObis = df.pivot_table(index=['newinput'],
                     columns=['newoutput'],
                     values=['invcoeff'])
df_invIObis = pd.DataFrame(np.linalg.inv(df_IObis), df_IObis.index, df_IObis.index).reset_index()

df = pd.melt(frame=df_invIObis,
              id_vars='newinput',
              var_name='newoutput',
              value_name='invindex')

df = df.groupby(['newinput']).agg(sum).reset_index()
df = df.rename(columns={'invindex': 'upstreamness',
                        'newinput': 'newind'})

# Merging with industry names 
df = pd.merge(df, df_io_newind_list, on='newind')
df = df[['io_industry', 'io_industry_name', 'upstreamness']].sort_values('io_industry')

# Save the file
# f_upstreamness_by_industry = 'upstreamness_by_industry.dta'
# df.to_stata(f_upstreamness_by_industry)









