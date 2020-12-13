#%% Funktionen und Datenauslese
import pandas as pd
from datetime import datetime 
import datetime as dt
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

#%% FUNCTION AND DEFINITION SECTION
class Game:
    def __init__(self):
        self.category   = []
        self.team       = []
        self.start      = []
        self.end        = []
    
    def readGameFromFrame(self, frame, language='willi', t0=[]):
        nEntries = len(frame)
        if language == 'willi':
            self.category   = np.empty_like(frame['Work Item'])
            self.team       = np.empty_like(frame['Work Item'])
            self.start      = np.empty_like(frame['Work Item'])
            self.end        = np.empty_like(frame['Work Item'])
        if not t0:
            t0 = self.trackerDate(frame['Start'][nEntries-1])
        for idi, rawCategory in enumerate(frame['Work Item']):
            cat, team           = self.splitWilli(rawCategory)
            self.category[idi]  = cat
            self.team[idi]      = team
            self.start[idi]     = (self.trackerDate(frame['Start'][idi]) - t0).total_seconds()
            self.end[idi]       = (self.trackerDate(frame['End'][idi]) - t0).total_seconds()

    def getCategories(self):
        uCats = np.unique(self.category)
        return uCats

    def getEntries(self, category, team):
        sorter = np.logical_and(self.category == category, self.team == team)
        starts = self.start[sorter]
        ends   = self.end[sorter]
        return starts, ends

    def getTimeFrame(self):
        return np.array([np.min(self.start), np.max(self.end)])

    def __str__(self):
        s = 'Category\t Team\t Start\t End\n'
        for ide in range(len(self.category)):
            sstr = str(dt.timedelta(seconds=self.start[ide]))
            estr = str(dt.timedelta(seconds=self.end[ide]))
            s = s + '%s\t\t %s\t %s \t %s\n' % (self.category[ide], self.team[ide], sstr, estr)
        return s
    def __repr__(self):
        return self.__str__()

    
    @staticmethod
    def trackerDate(datestr):
        return datetime.strptime(datestr, '%d.%m.%Y %H:%M:%S')

    @staticmethod
    def splitWilli(instr):
        sstr = instr.split('_')
        if len(sstr) == 3:
            cat     = sstr[1]
            team    = sstr[2]
        else:
            cat     = sstr[1]
            team    = 'referee'
        return cat, team


class IntervallAction:
    def __init__(self, start, end):
        self.start      = start
        self.end        = end
        self.duration   = end-start

    def __str__(self):
        s = ''
        for idi in range(0, len(self.start)):
            s += '%.4d\t %.4d\t %.4d\n' % (self.start[idi], self.end[idi], self.duration[idi])
        return s
    
    def __repr__(self):
        return self.__str__()

    def getXYTime(self, lower=0, upper=0):
        x = np.ones(len(self.start)*3)*lower
        y0 = np.ones_like(x)*lower
        y1 = np.ones_like(x)*upper
        w0 = np.ones_like(x)
        x[0::3] = self.start
        x[1::3] = self.end    
        x[2::3] = self.end
        w0[2::3] = 0
        return x, y0, y1, w0

    def plot(self, ax, color, label=[], lower=0, upper=1):
        x, y0, y1, w = self.getXYTime(lower=lower, upper=upper)
        if label==[]:
            ax.fill_between(x, y0, y1, where=w, color=color, edgecolor='none')
        else:
            ax.fill_between(x, y0, y1, where=w, color=color, label=label,edgecolor='none')


class Team:
    def __init__(self, name, identifier, color):
        self.name       = name
        self.identifier = identifier
        self.color      = color

        self.possession = []
        self.attack     = []
        self.scrum      = []

        self.goal       = []
        self.penalty    = []        
        self.freeThrow  = []
        self.passes     = []
        self.misspass   = []
            
    def setIntervallCategory(self, category, category_identifier, game):
        start, end = game.getEntries(category_identifier, self.identifier)
        setattr(self, category, IntervallAction(start, end))

    def setEventCategory(self, category, category_identifier, game):
        start, end = game.getEntries(category_identifier, self.identifier)
        setattr(self, category, start)

    def readWilliGame(self, game):
        iCats       = ['BB', 'TA', 'Scrum']
        iCatIdents  = ['possession', 'attack', 'scrum']
        eCats       = ['Unterbrechung']
        eCatIdents  = ['freeThrow']
        for idc, iCat in enumerate(iCats):
            self.setIntervallCategory(iCatIdents[idc], iCat, game)
        for ide, eCat in enumerate(eCats):
            self.setEventCategory(eCat, eCatIdents[ide], game)
        

def plotside(ax, h, width=0.4):
    ax.plot([0, 48*60000, 48*60000, 0], h + np.array([width, width, -width, -width]), 'w', linewidth=0.5, color='w')
#%% Read and prepare Data


# colors
bgc = np.array([1,1,1])*0
tcb = np.array([103,169,240])/255
tcw = np.array([1,1,1])*1
cg  = np.array([1,1,1])*0.3

iData   = pd.read_csv('col-nor.csv')
game    = Game()
game.readGameFromFrame(iData)
teams = [Team('NOR', 'weiss', tcw), Team('COL', 'blau', tcb), Team('REF', 'referee', [1., 0.3, 0.3])]
for team in teams:
    team.readWilliGame(game)


#%% Figure


fig, ax0 = plt.subplots(1, constrained_layout=True, figsize=(16,5))
fig.patch.set_facecolor(bgc)        # set background color to black
axs = [ax0.twiny(), ax0]

for ax in axs:
    t0, te = game.getTimeFrame()
    ax.set_xlim(t0, te)
    ax.set_facecolor(bgc)
    ax.xaxis.set_ticks(np.arange(t0, te, 5*60))
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
ax.set_xlabel('time (mm:ss, continuous)')


yticks = [0]
ylabel = ['              general']
ax.yaxis.set_ticks(yticks)
ax.set_yticklabels(ylabel)
ax.set_ylim([-1, 1])


cs      = [0.2, 0.35]               # center line to plot on
ls      = [0.1, 0.1]                # width around center line
cats    = ['possession', 'attack']
for idc in range(0, len(cs)):
    c   = cs[idc]
    l   = ls[idc]
    cat = cats[idc]
    getattr(teams[0], cat).plot(ax, tcb, label=cat, lower=c-l, upper=c+l)
    getattr(teams[1], cat).plot(ax, tcw, label=cat, lower=-c-l, upper=-c+l)

teams[2].scrum.plot(ax, teams[2].color, label='scrum', lower=-l, upper=l)


# %%
