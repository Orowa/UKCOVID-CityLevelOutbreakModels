import pandas as pd
import cufflinks as cf
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
from scipy.integrate import odeint
from IPython.display import display
cf.go_offline()
import re

class Model():
    
    def __init__(self, name, func):
        
        self.name = name
        self.func = func # function describing change in total cases as a function of (t, params, init conditions)
        
    def load(self, data, pop, city):
        
        self.raw_data = data # data should be a pandas df with columns Date and TotalCases
        self.pop = pop
        self.city = city
        
        # remove cases where incidence falls
        
        delta = data.TotalCases.diff()
        data = data[delta>=0]
        
        # remove pre-infection
        
        data = data[data.TotalCases>0]
        
        # add incidence data
        
        data['Incidence'] = data.TotalCases.diff()
        data['Incidence'].iloc[0] = 1 # first case must have a positive incidence

        # track days since day 0
        
        day0 = data['Date'].iloc[0]
        data['Time'] = [d.days for d in (data.Date - day0)]
        
        # save cleaned data
        
        self.data = data
    
    def popconvert(self, p):
        
        assert self.pop is not None, "Population not supplied"
        
        if type(p) is str:
            
            return float(re.findall("\d+\.\d+", p)[0])*self.pop
            
        return p
    
    def fit(self, par0, parlower, parupper):
        
        par0 = {key:self.popconvert(value) for (key,value) in par0.items()}
        parlower = {key:self.popconvert(value) for (key,value) in parlower.items()}
        parupper = {key:self.popconvert(value) for (key,value) in parupper.items()}
        
        self.par0 = par0
        self.parbounds = (parlower, parupper)
        
        if self.data is None:
                
            print("No data loaded to fit.")
            return None
        
        params, _ = opt.curve_fit(self.func,
                                 self.data.Time,
                                 self.data.TotalCases,
                                 p0=list(par0.values()),
                                 bounds=(list(parlower.values()),list(parupper.values())))
        
        self.params = dict(zip(par0.keys(),params))
        
    def fit_bootstrap(self, par0, parlower, parupper, S=100):
        
        self.fit(par0, parlower, parupper)
        
        # Get predicted daily incidents from best fit
        
        periods = self.data.Time.iloc[-1]
        best_fit = self.func(range(periods), **self.params)
        best_daily = np.diff(best_fit)
        
        # Initialise
        
        paramouts = []
        
        # Bootstrap
        
        for i in range(S):

            new_daily = np.array([np.random.poisson(lam) for lam in best_daily])
            new_cum = np.insert(new_daily,0,1).cumsum() # account for the fact that first case must exist

            try:
                
                param_new, _ = opt.curve_fit(self.func,
                                 range(periods),
                                 new_cum,
                                 p0=list(self.par0.values()),
                                 bounds=(list(self.parbounds[0].values()),list(self.parbounds[1].values())))

                paramouts.append(param_new)

            except:

                continue
                
        # Summarise
        
        pars_df = pd.DataFrame(paramouts)
        
        params_sd = pars_df.std(0)
        params_lower = pars_df.quantile(.25)
        params_upper = pars_df.quantile(.75)
        
        self.params_sd = dict(zip(self.par0.keys(),params_sd))
        self.params_lower = dict(zip(self.par0.keys(),params_lower))
        self.params_upper = dict(zip(self.par0.keys(),params_upper))
        
        
    def visualise(self, lookahead = 7, prediction = True, intervals = True):
        
        fig = self.data.iplot(x='Date', y='TotalCases',
                 mode = 'lines+markers',
                 asFigure = True, 
                 title="Case count "+self.city,
                 yTitle="Total cases",
                 xTitle="Date",
                 color="orange")
        
        fig.add_trace(go.Bar(x=self.data.Date, y=self.data.Incidence, name='DailyIncidents'))

        
        if prediction:
            
            if self.params is None:
                
                print("No parameters fitted.")
                return None
            
            periods = self.data.Time.iloc[-1] + 1 + lookahead
            best_fit = self.func(range(periods), **self.params)
            best_daily = np.diff(best_fit)
            
            dates = pd.date_range(start=self.data.Date.iloc[0],periods=periods)
            
            fig.add_trace(go.Scatter(x=dates, y=best_fit, line=dict(color="green"), name='EstTotalCases ('+self.name+')'))
            fig.add_trace(go.Bar(x=dates, y=best_daily, marker=dict(color="blue"), name='EstDailyIncidents ('+self.name+')'))

            if intervals:
                
                if self.params_sd is None:
                    
                    print("No confidence intervals fitted")
                    return None
                
                upper_params = {key:(min(self.params[key]+self.params_sd[key],self.parbounds[1][key])) for key in self.params.keys()}
                lower_params = {key:(max(self.params[key]-self.params_sd[key],self.parbounds[0][key])) for key in self.params.keys()}
                
                
                
                best_fit_upper = self.func(range(periods), **upper_params)
                best_fit_lower = self.func(range(periods), **lower_params)
                
                fig.add_trace(go.Scatter(x=dates, y=best_fit_upper, line=dict(color="green", dash="dash"), name='EstTotalCases ('+self.name+', upper)'))
                fig.add_trace(go.Scatter(x=dates, y=best_fit_lower, line=dict(color="green", dash="dash"), name='EstTotalCases ('+self.name+', lower)'))


        fig.show()
        
    def assess(self, lookaheads=7, visualise=True, inplace=True):
        
        rmses = []

        for lookahead in range(1,lookaheads):

            if (len(self.data)-lookahead) <= 8:

                print("Not enough data to assess forecast of "+str(lookahead)+" periods")
                continue

            target_dates = self.data.Date.iloc[7:(-lookahead)]
            targets = self.data.TotalCases.iloc[7:(-lookahead)].values
            preds = []

            for t in target_dates:

                train = self.data[self.data.Date < t]

                param_new, _ = opt.curve_fit(self.func,
                                     train.Time,
                                     train.TotalCases,
                                     p0=list(self.par0.values()),
                                     bounds=(list(self.parbounds[0].values()),list(self.parbounds[1].values())))

                params_dict = dict(zip(self.par0.keys(),param_new))

                predict = self.func(range(train.Time.iloc[-1]+lookahead+1),**params_dict)
                preds.append(predict[-1])

            rmse = np.sqrt(np.mean((np.array(preds)-targets)**2))
            rmses.append(rmse)

        if visualise:

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1,lookaheads)), y=rmses))
            fig.update_layout(
                title="RMSE per forecast length",
                xaxis_title="Forecast length (days)",
                yaxis_title="RMSE (for cumulative cases that many days ahead)")
            fig.show()
        
        if inplace:
            
            self.rmses = rmses
        
        else:
            
            return rmses