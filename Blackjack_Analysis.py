# -*- coding: utf-8 -*-
"""
@author: Colton Smith
"""

#http://pydealer.readthedocs.io/en/latest/index.html

import pydealer
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

deck = pydealer.Deck()
pen = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
multi = [1,2,3,4,5,6,7,8]
sims = 10000

def new_shoe(m_s):
    shoe = pydealer.Stack()
    for i in range(0,m_s):
        shoe.add(deck)
    shoe.shuffle()
    return shoe

counts = {
    "Ace": -1,
    "King": -1,
    "Queen": -1,
    "Jack": -1,
    "10": -1,
    "9": 0,
    "8": 0,
    "7": 0,
    "6": 1,
    "5": 1,
    "4": 1,
    "3": 1,
    "2": 1
}

### Save Parameter Combination Results ###
s_pen = []
s_multi = []
s_points = []
s_time = []
s_var = []

for m in range(0,len(multi)):
    for p in range(0,len(pen)):
        m_s = multi[m]
        p_s = pen[p]
        points = []
        time = []
        var = []   
        for i in range(0,sims):
            rcount_hist = []
            tcount_hist = []
            rcount = 0
            tcount = 0
            shoe = new_shoe(m_s)
            
            cut = int(round(shoe.size*p_s,0))
            for j in range(0,cut):
                hand = shoe.deal(1)
                card = hand[0].value
                c_count = counts.get(card)
                rcount += c_count
                if (shoe.size == 0):
                    tcount = 0
                else:
                    tcount = rcount / (shoe.size/deck.size)
                rcount_hist.append(rcount)
                tcount_hist.append(tcount)
        
            favor = len([k for k in tcount_hist if k > 2])
            points.append(favor)    
            time.append((favor/len(tcount_hist))*100) 
            var.append(np.std(tcount_hist))

#            plt.plot(tcount_hist)
#            plt.title(str(m_s) + ' Decks with ' + str(p_s) + ' Penetration')
#            plt.xlabel('Cards Dealt')
#            plt.ylabel('True Count')
        
        s_pen.append(p_s)
        s_multi.append(m_s)
        s_points.append(np.mean(points))
        s_time.append(np.mean(time))
        s_var.append(np.mean(var))



### Plot Heatmaps of Parameters ###

results = pd.DataFrame(
    {'Number of Decks': s_multi,
     'Penetration': s_pen,
     'Advantage_Points': s_points,
     'Advantage_Time': s_time,
     'Variance': s_var
    })

    
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=False, squeeze=True)
    
adv_h = results[['Number of Decks','Penetration','Advantage_Points']]
adv_h = adv_h.pivot(index='Number of Decks', columns='Penetration', values='Advantage_Points')
fig1 = sb.heatmap(adv_h, annot = True, cbar = False, cmap = 'RdYlGn', ax = axes[0])
axes[0].set_title('Number of Points with TC > 2 during ' + str(sims) + ' Simulations')

adv_p = results[['Number of Decks','Penetration','Advantage_Time']]
adv_p = adv_p.pivot(index='Number of Decks', columns='Penetration', values='Advantage_Time')
fig2 = sb.heatmap(adv_p, annot = True, cbar = False, cmap = 'RdYlGn', ax = axes[1], yticklabels = False)
axes[1].set_title('Percent of Shoe with TC > 2 during ' + str(sims) + ' Simulations')
axes[1].set_ylabel('')    
axes[1].set_xlabel('')

flip = results[['Number of Decks','Penetration','Variance']]
flip = flip.pivot(index='Number of Decks', columns='Penetration', values='Variance')
fig3 = sb.heatmap(flip, annot = True, cbar = False, cmap = 'RdYlGn', ax = axes[2], yticklabels = False)
axes[2].set_title('Standard Deviation of the True Count during ' + str(sims) + ' Simulations') 
axes[2].set_ylabel('')    
axes[2].set_xlabel('')

