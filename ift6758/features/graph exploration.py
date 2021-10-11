# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 10:07:11 2021

@author: Marc-André
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


DATA_DIR = os.path.join('..', 'data')

tiny = pd.read_csv(os.path.join(DATA_DIR, 'tidy_data.csv'))
tiny =tiny.dropna(subset = ["x_coordinate"]) #need to remove NaN


tiny['x_coordinate_adj'] = np.where(tiny["x_coordinate"]<0, -tiny["x_coordinate"], tiny["x_coordinate"] )
tiny['y_coordinate_adj'] = np.where(tiny["x_coordinate"]<0, -tiny["y_coordinate"]+42, tiny["y_coordinate"]+42 )

#year hard coded for now
def compute_league_avg(df, year=2017):
    df_copy = df[df["season"] ==20172018].copy()
    df_copy["coord_tuple"] = df_copy[["x_coordinate_adj","y_coordinate_adj"]].apply(tuple, axis=1)
    
    data_league= np.zeros((100,85))
    
    for i,j in df_copy['coord_tuple']:
        data_league [int(i),int(j)] += 1
    
    data_league = data_league/2542 #need to count each game twice since two team, need to replace with actual calculation of total game time
    
    #freq_dist = df_copy[["x_coordinate","y_coordinate"]].value_counts().reset_index(name="freq_league")
    #freq_dist["freq_league"] = freq_dist["freq_league"]/1271
    return data_league

#team and year hard coded for now
def compute_team_avg(df, year=2017, team = "Colorado Avalanche"):
    df_copy = df[df["season"] ==20172018].copy()
    df_copy2 = df_copy[df_copy['team']==team].copy()
    df_copy2["coord_tuple"] = df_copy2[["x_coordinate_adj","y_coordinate_adj"]].apply(tuple, axis=1)
    
    data_team = np.zeros((100,85))
    
    for i,j in df_copy2['coord_tuple']:
        data_team[int(i),int(j)] += 1
    
    data_team = data_team/82 #need to replace 82 with real time calculation
    
    return data_team


test_league = compute_league_avg(tiny)

test_team = compute_team_avg(tiny)


test_total = test_team - test_league
test_total = gaussian_filter(test_total, sigma=4) #smoothing results


xx, yy = np.mgrid[0:100:100j, -42:42:85j]

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(0, 100)
ax.set_ylim(-42, 42)
#plots
cfset = ax.contourf(xx, yy, test_total, cmap='bwr')
cset = ax.contour(xx, yy, test_total, colors='k')
# Label plots
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(cfset)
cfset.set_clim(-0.03,0.03)#need to find a way to centralise white = 0
plt.show()
