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
    x = np.ones(len(start)*3)*lower
    y0 = np.ones_like(x)*lower
    y1 = np.ones_like(x)*upper
    w0 = np.ones_like(x)
    x[0::3] = start
    x[1::3] = stop    
    x[2::3] = stop
    w0[2::3] = 0
    return x, y0, y1, w0


def plotTimeXY(start, stop, ax, color, label=[], lower=0, upper=1):
    x, y0, y1, w = getTimeXY(start, stop, lower=lower, upper=upper)
    if label==[]:
        ax.fill_between(x*1e3, y0, y1, where=w, color=color, edgecolor='none')
    else:
        ax.fill_between(x*1e3, y0, y1, where=w, color=color, label=label,edgecolor='none')


def plotside(ax, h, width=0.4):
    ax.plot([0, 48*60000, 48*60000, 0], h + np.array([width, width, -width, -width]), 'w', linewidth=0.5, color='w')

a = dict()
b = dict()
w = dict()

a['scrum']      = importTag(r'scrum.xlsx')
a['timeout']    = importTag(r'timeout.xlsx')

b['Ballbesitz'] = importTag(r'ballbesitzb.xlsx')
b['Tor']        = importTag(r'torb.xlsx')
b['Freiwurf']   = importTag(r'freib.xlsx')
b['Torangriff']    = importTag(r'angriffb.xlsx')
b['Strafzeit']    = importTag(r'strafzeitb.xlsx')

w['Ballbesitz'] = importTag(r'ballbesitzw.xlsx')
w['Pass']       = importTag(r'passw.xlsx')
w['Tor']        = importTag(r'torw.xlsx')
w['Freiwurf']   = importTag(r'freiw.xlsx')
w['Torangriff']    = importTag(r'angriffw.xlsx')
w['Strafzeit']    = importTag(r'strafzeitw.xlsx')

#%% Abbildung
bgc = np.array([1,1,1])*0
tcb = np.array([103,169,240])/255
tcw = np.array([1,1,1])*1
cg  = np.array([1,1,1])*0.3
fig, ax0 = plt.subplots(1, constrained_layout=True, figsize=(16,5))
fig.patch.set_facecolor(bgc)
axs = [ax0.twiny(), ax0]

for ax in axs:
    ax.set_facecolor(bgc)
    ax.xaxis.set_ticks(np.arange(5*6e4, 50*60000, 5*60000))
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s // 1000)))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(0, 45*60000)
ax.set_xlabel('Zeit (mm:ss, laufend)')


yticks = [0]
ylabel = ['              Allgemeines']
ax.yaxis.set_ticks(yticks)
ax.set_yticklabels(ylabel)
ax.set_ylim([-1, 1])
c = 0; l = 0.1; j = 0
plotside(ax, 0, width=l)
ax.plot(np.ones(2)*(18*6e4+50e3), [-1, +1], '--', linewidth=3, label='Halbzeitpause', color=[.5,.5,.5])
ax.plot(1e3*b['Tor'][0], np.ones_like(b['Tor'][0])*c, '*', color=tcb, markersize=20, markeredgecolor=cg)
ax.plot(1e3*w['Tor'][0], np.ones_like(w['Tor'][0])*c, '*', color='w', markersize=20, markeredgecolor=cg, label='Tor')
ax.legend(ncol=5, loc=3)
fig.savefig('analyse%d.png' % j, facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1

teams = [b, w]
color = [tcb, tcw]
label = ['blau', 'weiß']

cgb     = np.array([66, 201, 54])/255
cs      = [0.2, 0.35, 0.5]
ls      = [0.1, 0.1, 0.05]
cats    = ['Ballbesitz', 'Torangriff', 'Strafzeit']
ccolw   = [tcw, cgb, 'r']
ccolb   = [tcb, cgb, 'r']


for i, cat in enumerate(cats):
    c = cs[i]
    l = ls[i]
    plotTimeXY(b[cat][0], b[cat][1], ax=ax, color=ccolb[i], lower=c-l, upper=c+l)
    plotTimeXY(w[cat][0], w[cat][1], ax=ax, color=ccolw[i], lower=-c-l, upper=-c+l)
    #plotside(ax, c, width=l)
    #plotside(ax, -c, width=l)
    yticks.append(c)
    yticks.append(-c)
    ylabel.append('%s (DEN)' % cat)
    ylabel.append('%s (GER)' % cat)
    ax.yaxis.set_ticks(yticks)
    ax.set_yticklabels(ylabel)
    ax.set_ylim([-1, 1])
    fig.savefig('analyse%d.png' % j, facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1

c = 0; l = 0.1
ax.plot(1e3*b['Freiwurf'][0], np.ones_like(b['Freiwurf'][0])*c, '>', color=tcb, markersize=10, markeredgecolor=cg)
ax.plot(1e3*w['Freiwurf'][0], np.ones_like(w['Freiwurf'][0])*c, '>', color='w', markersize=10, markeredgecolor=cg, label='Freiwurf')
ax.legend(ncol=5, loc=3)
fig.savefig('analyse%d.png' % j, facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1



plotTimeXY(a['timeout'][0], a['timeout'][1], ax=ax, color=[0.5, .5, .5], label='Auszeit', lower=c-l, upper=c+l)
plotTimeXY(a['scrum'][0],   a['scrum'][1],   ax=ax, color=[1., 0.3, 0.3], label='Gerangel', lower=c-l, upper=c+l)
ax.legend(ncol=5, loc=3)
fig.savefig('analyse%d.png' % j, facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1





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
# %% Tor 1
fig, ax0 = plt.subplots(1, constrained_layout=True, figsize=(16,5))
ax = ax0
fig.patch.set_facecolor(bgc)
ax.set_facecolor(bgc)
ax.xaxis.set_ticks(np.arange(0*6e4, 50*60000, 2.5*60000))
formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s // 1000)))
ax.xaxis.set_major_formatter(formatter)
ax.set_xlim(0, 45*60000)
ax.set_xlabel('Zeit (mm:ss, laufend)')


yticks = [0]
ylabel = ['              Allgemeines']
ax.yaxis.set_ticks(yticks)
ax.set_yticklabels(ylabel)
ax.set_ylim([-1, 1])
c = 0; l = 0.1; j = 0
plotside(ax, 0, width=l)
ax.plot(np.ones(2)*(18*6e4+50e3), [-1, +1], '--', linewidth=3, label='Halbzeitpause', color=[.5,.5,.5])
ax.plot(1e3*b['Tor'][0], np.ones_like(b['Tor'][0])*c, '*', color=tcb, markersize=20, markeredgecolor=cg)
ax.plot(1e3*w['Tor'][0], np.ones_like(w['Tor'][0])*c, '*', color='w', markersize=20, markeredgecolor=cg, label='Tor')
ax.legend(ncol=5, loc=3)

teams = [b, w]
color = [tcb, tcw]
label = ['blau', 'weiß']

cgb     = np.array([66, 201, 54])/255
cs      = [0.2, 0.35]
ls      = [0.1, 0.1]
cats    = ['Ballbesitz', 'Torangriff']
ccolw   = [tcw, cgb]
ccolb   = [tcb, cgb]


for i, cat in enumerate(cats):
    c = cs[i]
    l = ls[i]
    plotTimeXY(b[cat][0], b[cat][1], ax=ax, color=ccolb[i], lower=c-l, upper=c+l)
    plotTimeXY(w[cat][0], w[cat][1], ax=ax, color=ccolw[i], lower=-c-l, upper=-c+l)
    #plotside(ax, c, width=l)
    #plotside(ax, -c, width=l)
    yticks.append(c)
    yticks.append(-c)
    ylabel.append('%s (DEN)' % cat)
    ylabel.append('%s (GER)' % cat)
    ax.yaxis.set_ticks(yticks)
    ax.set_yticklabels(ylabel)
    ax.set_ylim([-1, 1])
ax.set_xlim([4*6e4, 6e4*11])

fig.savefig('analyse_tor11.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1

c = 0; l = 0.1
ax.plot(1e3*w['Pass'][0], np.ones_like(w['Pass'][0])*c, '+', color='w', markersize=10, label='Pass')
ax.legend(ncol=5, loc=3)
fig.savefig('analyse_tor12.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1

ax.plot(1e3*b['Freiwurf'][0], np.ones_like(b['Freiwurf'][0])*c, '>', color=tcb, markersize=10, markeredgecolor=cg)
ax.plot(1e3*w['Freiwurf'][0], np.ones_like(w['Freiwurf'][0])*c, '>', color='w', markersize=10, markeredgecolor=cg, label='Freiwurf')
plotTimeXY(a['timeout'][0], a['timeout'][1], ax=ax, color=[0.5, .5, .5], label='Auszeit', lower=c-l, upper=c+l)
plotTimeXY(a['scrum'][0],   a['scrum'][1],   ax=ax, color=[1., 0.3, 0.3], label='Gerangel', lower=c-l, upper=c+l)
ax.legend(ncol=5, loc=3)

fig.savefig('analyse_tor13.png', facecolor=fig.get_facecolor(), transparent=True, dpi=150); j=j+1


# %%
