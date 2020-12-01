#%% Funktionen und Datenauslese
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import time


params = {"ytick.color" : "w",
          "xtick.color" : "w",
          "axes.labelcolor" : "w",
          "axes.edgecolor" : "w"}
plt.rcParams.update(params)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

def importTag(fName, t0   = datetime(1900, 1, 1, 0, 0, 0, 0)):
    dframe  = pd.read_excel(fName)
    cols    = pd.DataFrame(dframe, columns=['Beginning', 'End'])
    start   = np.zeros(len(cols['Beginning']))
    end     = np.zeros(len(cols['Beginning']))
    for idc in range(0, len(cols)):
        start[idc] = (datetime.strptime(cols['Beginning'][idc], '%H:%M:%S,%f') - t0).total_seconds()
        end[idc]   = (datetime.strptime(cols['End'][idc], '%H:%M:%S,%f') - t0).total_seconds()
    return start, end

def getTimeXY(start, stop, lower=0, upper=1):
    x = np.ones(len(start)*4)*lower
    y = np.ones(len(start)*4)*lower
    x[0::4] = start
    x[1::4] = start
    x[2::4] = stop
    x[3::4] = stop
    y[1::4] = upper
    y[2::4] = upper
    return x, y


def plotTimeXY(start, stop, ax, color, label=[], lower=0, upper=1):
    x, y = getTimeXY(start, stop, lower=lower, upper=upper)
    ax.fill_between(x*1e3, lower, y, color=color, label=label)


def plotside(ax, h, width=0.4):
    ax.plot([0, 48*60000, 48*60000, 0], h + np.array([width, width, -width, -width]), 'k-', linewidth=1.5, color='w')

a = dict()
b = dict()
w = dict()

a['scrum']      = importTag(r'scrum.xlsx')
a['timeout']    = importTag(r'timeout.xlsx')

b['ballbesitz'] = importTag(r'ballbesitzb.xlsx')
b['tor']        = importTag(r'torb.xlsx')
b['freiwurf']   = importTag(r'freib.xlsx')
b['angriff']    = importTag(r'angriffb.xlsx')

w['ballbesitz'] = importTag(r'ballbesitzw.xlsx')
w['pass']       = importTag(r'passw.xlsx')
w['tor']        = importTag(r'torw.xlsx')
w['freiwurf']   = importTag(r'freiw.xlsx')
w['angriff']    = importTag(r'angriffw.xlsx')

#%% Abbildung
bgc = np.array([1,1,1])*0
tcb = np.array([103,169,240])/255
tcw = np.array([1,1,1])*1
cg  = np.array([1,1,1])*0.2
fig, ax0 = plt.subplots(1, constrained_layout=True, figsize=(16,5))
fig.patch.set_facecolor(bgc)
axs = [ax0.twiny(), ax0]

h0 = 7.
h = h0

for ax in axs:
    ax.set_facecolor(bgc)
    ax.yaxis.set_ticks(np.arange(1, h0+1., 1.))
    #ax.set_yticklabels(['', '', 'Strafzeiten', 'Freiwürfe', 'Ballbesitz', 'Torangriffe', 'Tore\nPässe'])
    #ax.set_ylim([0, h0+1])
    ax.xaxis.set_ticks(np.arange(5*6e4, 50*60000, 5*60000))
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s // 1000)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(0, 45*60000)

ax.set_xlabel('Zeit (mm:ss, laufend)')




##
# Tore und Pässe
plt.plot(1e3*w['pass'][0], np.ones_like(w['pass'][0])*h, '+', color=tcw, markersize=15)
plt.plot(1e3*b['tor'][0], np.ones_like(b['tor'][0])*h, '*', color=tcb, markersize=30, markeredgecolor=cg)
plt.plot(1e3*w['tor'][0], np.ones_like(w['tor'][0])*h, '*', color=tcw, markersize=30, markeredgecolor=cg)
h = h-1

plotTimeXY(b['angriff'][0], b['angriff'][1], ax=ax, color=tcb, lower=h-0.4, upper=h+0.4)
plotTimeXY(w['angriff'][0], w['angriff'][1], ax=ax, color=tcw, lower=h-0.4, upper=h+0.4)
plotside(ax, h)


h = h-.8
plotTimeXY(b['ballbesitz'][0], b['ballbesitz'][1], ax=ax, color=tcb, label='blau', lower=h-0.1, upper=h+0.1)
plotside(ax, h, width=0.1)
h = h - 0.2
plotTimeXY(a['timeout'][0], a['timeout'][1], ax=ax, color=[0.5, .5, .5], label='Auszeit', lower=h-0.1, upper=h+0.1)
plotTimeXY(a['scrum'][0], a['scrum'][1], ax=ax, color=[1., 0.3, 0.3], label='Gerangel', lower=h-0.1, upper=h+0.1)
plotside(ax, h, width=0.1)
h = h - 0.2
plotTimeXY(w['ballbesitz'][0], w['ballbesitz'][1], ax=ax, color=tcw, label='weiß', lower=h-0.1, upper=h+0.1)
plotside(ax, h, width=0.1)
h = h - 1

ax.plot(np.ones(2)*(18*6e4+50e3), [0, h0+4], 'w--', linewidth=2)
ax.legend(ncol=4, loc=9)

fig.savefig('chart.png', facecolor=fig.get_facecolor(), transparent=True)
#%% Statistiken
bbw_g = np.sum(b['ballbesitz'][1] - b['ballbesitz'][0])
bbb_g = np.sum(w['ballbesitz'][1] - w['ballbesitz'][0])
bb_total = bbb_g + bbw_g
fb = len(b['freiwurf'][0])
fw = len(w['freiwurf'][0])

ab = len(b['angriff'][0])
aw = len(w['angriff'][0])

mab = np.mean(np.diff(b['angriff'], axis=0))
maw = np.mean(np.diff(w['angriff'], axis=0))

print('Dänemark \t\t\t Deutschland')
print('%d   \t\t Pässe \t %d' % (0, len(w['pass'][0])))
print('%.1f \t\t Ballbesitz \t %.1f' % (bbb_g/bb_total*100, bbw_g/bb_total*100))
print('%d   \t\t Angriffe \t %d' % (ab, aw))
print('%.1f \t\t Angriffsdauer \t %.1f' % (mab, maw))
print('%d   \t\t Freiwürfe \t %d' % (fb, fw))
# %%
