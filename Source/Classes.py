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
    
    def __init__(self, name, func, par0=None, parupper=None, parlower=None):
        
        self.name = name
        self.func = func # function describing change in total cases as a function of (t, params, init conditions)
        
        self.par0 = par0
        self.parbounds = (parlower, parupper)
                
    def load(self, data, pop, city):
        
        self.raw_data = data # data should be a pandas df with columns Date and TotalCases
        self.pop = pop
        self.city = city
        
        # remove pre-infection
        
        data = data[data.TotalCases>0]
        
        # remove cases where incidence falls
        
        def monotonic(data):
    
            pmax = 0
            keep = []

            for i in range(len(data)):

                keep.append(data[i]>=pmax)
                pmax = max(pmax, data[i])

            return keep
        
        keep = monotonic(data.TotalCases.values)
        data = data[keep]

        # add incidence data

        data['Incidence'] = data.TotalCases.diff()
        data['Incidence'].iloc[0] = data['TotalCases'].iloc[0] # first case must have a positive incidence

        # track days since day 0
        
        day0 = data['Date'].iloc[0]
        data['Time'] = [d.days for d in (data.Date - day0)]
        
        # save cleaned data
        
        self.data = data
        self.C_0 = data.TotalCases.iloc[0]
    
    def popconvert(self, p):
        
        assert self.pop is not None, "Population not supplied"
        
        if type(p) is str:
            
            return float(re.findall("\d+\.\d+", p)[0])*self.pop
            
        return p
    
    def fit(self, par0 = None, parlower = None, parupper = None):
        
        # If new parameter vectors provided, override the default params:
        
        if par0:
            self.par0 = par0
        if parlower or parupper:
            self.parbounds = (parlower, parupper)
            
        # Sense check data and parameters have actually been provided
        
        if self.data is None:
                
            print("No data loaded to fit.")
            return None
        
        if self.par0 is None:
            
            print("No parameters available.")
            return None
            
        if self.parbounds is None:
            
            print("No parameter bounds available.")
            return None
        
        # Realise the actual default parameters by subsituting in the population variables
            
        self.par0_actual = {key:self.popconvert(value) for (key,value) in self.par0.items()}
        parlower = {key:self.popconvert(value) for (key,value) in self.parbounds[0].items()}
        parupper = {key:self.popconvert(value) for (key,value) in self.parbounds[1].items()}
        self.parbounds_actual = (parlower, parupper)
        
        # Fit the parameters
        
        params, _ = opt.curve_fit(self.func,
                                  self.data.Time,
                                  self.data.TotalCases,
                                  p0=list(self.par0_actual.values()),
                                  bounds=(list(self.parbounds_actual[0].values()),list(self.parbounds_actual[1].values())))
        
        self.params = dict(zip(self.par0_actual.keys(),params)) # best fit parameters
        
    def fit_bootstrap(self, par0 = None, parlower = None, parupper = None, S=50):
        
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
                                 p0=list(self.par0_actual.values()),
                                 bounds=(list(self.parbounds_actual[0].values()),list(self.parbounds_actual[1].values())))

                paramouts.append(param_new)

            except:

                continue
                
        # Summarise
        
        pars_df = pd.DataFrame(paramouts)
        
        params_sd = pars_df.std(0)
        params_lower = pars_df.quantile(.25)
        params_upper = pars_df.quantile(.75)
        
        self.params_sd = dict(zip(self.par0.keys(),params_sd)) # standard deviation for realised bootstrapped parameters
        self.params_lower = dict(zip(self.par0.keys(),params_lower)) # lower bound for realised bootstrapped parameters
        self.params_upper = dict(zip(self.par0.keys(),params_upper)) # upper bound for realised bootstrapped parameters
        
        
    def visualise(self, lookahead = 7, prediction = True, intervals = True, intervalinit = False):
        
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
               

                # calculate the upper and lower parameter estimates, ensuring they are bounded appropriately by the starting bounds on the pars

                parlower = {key:self.popconvert(value) for (key,value) in self.parbounds[0].items()}
                parupper = {key:self.popconvert(value) for (key,value) in self.parbounds[1].items()}
                self.parbounds_actual = (parlower, parupper)
                
                upper_params = {key:(min(self.params[key]+self.params_sd[key],self.parbounds_actual[1][key])) for key in self.params.keys()}
                lower_params = {key:(max(self.params[key]-self.params_sd[key],self.parbounds_actual[0][key])) for key in self.params.keys()}
                
                if intervalinit:
                    
                    # fit the confidence intervals starting from day 1 of the outbreak

                    best_fit_upper = self.func(range(periods), **upper_params)
                    best_fit_lower = self.func(range(periods), **lower_params)
                
                    fig.add_trace(go.Scatter(x=dates, y=best_fit_upper, line=dict(color="green", dash="dash"), name='EstTotalCases ('+self.name+', upper)'))
                    fig.add_trace(go.Scatter(x=dates, y=best_fit_lower, line=dict(color="green", dash="dash"), name='EstTotalCases ('+self.name+', lower)'))

                else:
                        
                    C_t_est = best_fit[self.data.Time.iloc[-1]]

                    best_fit_upper = self.func(range(lookahead+1), **upper_params, C_0 = C_t_est) # mini hack - assumes the iniitial conditions for the dynamics function are called C_0
                    best_fit_lower = self.func(range(lookahead+1), **lower_params, C_0 = C_t_est)

                    dates = pd.date_range(start=self.data.Date.iloc[-1], periods=lookahead)

                    fig.add_trace(go.Scatter(x=dates, y=best_fit_upper, line=dict(color="green", dash="dash"), name='EstTotalCases ('+self.name+', upper)'))
                    fig.add_trace(go.Scatter(x=dates, y=best_fit_lower, line=dict(color="green", dash="dash"), name='EstTotalCases ('+self.name+', lower)'))

        fig.show()
        
    def assess(self, lookaheads=7, visualise=True, inplace=True):
        
        rmses = []

        for lookahead in range(1,lookaheads):

            if (len(self.data)-lookahead) <= 8:

                print("Not enough data to assess forecast of "+str(lookahead)+" periods")
                continue

            target_dates = self.data.Date.iloc[7:(-lookahead)].values
            targets = self.data.TotalCases.iloc[7:(-lookahead)].values
            
            preds = []
            final_targets = []
            
            for i in range(len(target_dates)):
                
                t = target_dates[i]
                
                train = self.data[self.data.Date < t]
                
                try:
                
                    param_new, _ = opt.curve_fit(self.func,
                                         train.Time,
                                         train.TotalCases,
                                         p0=list(self.par0_actual.values()),
                                         bounds=(list(self.parbounds_actual[0].values()),list(self.parbounds_actual[1].values())))

                    params_dict = dict(zip(self.par0.keys(),param_new))

                    predict = self.func(range(train.Time.iloc[-1]+lookahead+1),**params_dict)
                    
                    preds.append(predict[-1])
                    final_targets.append(targets[i])
                
                except:
                    
                    continue

            rmse = np.sqrt(np.mean((np.array(preds)-np.array(final_targets))**2))
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
        
class epidemic():
    
    def __init__(self, casedata, popdata):
        
        self.casedata = casedata
        self.popdata = popdata
        
    def fit_target(self, target, model,
                   intervals=False, visualise=False, assess=False, lookahead=7, inplace=True):
        
        try:
            
            pop = self.popdata.loc[self.popdata.Name == target,'Population'].iat[0]
                        
        except:
            
            print("Data not found for target city.")
            return None
        
        data = self.casedata[self.casedata.Area == target][['Date','TotalCases']]
        
        casedata = sum(data.TotalCases>0)
        
        if casedata<=7:
            
            print("Insufficient case data found for "+target+". Days with available case data = "+str(casedata)+".")
            return None
        
        print("Data found for "+target+", processing.")
        
        model.load(data, pop=pop, city=target)
        
        if intervals:
            if self.all_fitted_bootstrap:
                parnames = list(model.par0.keys())
                parnames_se = [x + '_se' for x in parnames]
                params = self.popdata.loc[self.popdata.Name == target, parnames].values[0]
                params_sd = self.popdata.loc[self.popdata.Name == target, parnames_se].values[0]
                model.params = dict(zip(parnames,params))
                model.params_sd = dict(zip(parnames, params_sd))
            else:    
                print("Bootstrapping. This may take some time.")
                model.fit_bootstrap(S=50)
            if visualise:
                model.visualise(lookahead=lookahead, intervals=True)
            if assess:
                if self.all_fitted_bootstrap:
                    # hack: the assess method needs the fit to have run beforehand to set the bound.
                    # todo: fix assess method to work with cache data
                    model.fit()
                model.assess(inplace=inplace)
            if not inplace:
                return model.params, model.params_sd
        
        else:
            if self.all_fitted:
                parnames = list(model.par0.keys())
                params = self.popdata.loc[self.popdata.Name == target, parnames].values[0]
                model.params = dict(zip(model.par0.keys(),params))
            else:
                model.fit()
            if visualise:
                model.visualise(lookahead=lookahead, intervals=False)
            if assess:
                if self.all_fitted:
                    # hack: the assess method needs the fit to have run beforehand to set the bound.
                    # todo: fix assess method to work with cache data
                    model.fit()
                model.assess(inplace=inplace)
            if not inplace:
                return model.params
        
#    def fit_all(self, model, verbose=True):
#        
#        target_codes = self.popdata.Code.values
#        self.popdata = pd.concat([self.popdata,pd.DataFrame(columns=list(model.par0.keys()))])
#        self.popdata['LatestTotalCases'] = np.nan
#        
#        for tc in target_codes:
#            
#            target = self.popdata.loc[self.popdata.Code == tc,'Name'].iat[0]
#            pop = self.popdata.loc[self.popdata.Code == tc,'Population'].iat[0]
#            data = self.casedata[self.casedata.AreaCode == tc][['Date','TotalCases']]
#            
#            if sum(data.TotalCases>0)<=7:
#                continue
#            
#            data = data.sort_values(by=['Date'], ascending = True)
#            self.popdata.loc[self.popdata.Code==tc, 'LatestTotalCases'] = data.TotalCases.values[-1]
#            
#            model.load(data, pop, target)
#            
#            try:
#                model.fit()
#                self.popdata.loc[self.popdata.Code==tc, model.params.keys()] = model.params.values()
#                if verbose:
#                    print(target, model.params)
#        
#            except BaseException as e:
#                print("Fit failed for "+target+"due to the following error:\n"+str(e))
#                continue
        
    def fit_all(self, model, verbose=True, bootstrap=False, dropna=True):
        
        target_codes = self.popdata.Code.values
        parnames = list(model.par0.keys())
        parnames_se = [x + '_se' for x in parnames]
        self.popdata = pd.concat([self.popdata,pd.DataFrame(columns=parnames+parnames_se)])
        self.popdata['LatestTotalCases'] = np.nan
        
        totalcases = len(target_codes)
        
        if verbose:
        
            print("Total cities to attempt: "+str(totalcases))
        
        attempts = 0
        successes = 0
        
        for tc in target_codes:
            
            target = self.popdata.loc[self.popdata.Code == tc,'Name'].iat[0]
            pop = self.popdata.loc[self.popdata.Code == tc,'Population'].iat[0]
            data = self.casedata[self.casedata.AreaCode == tc][['Date','TotalCases']]
            
            attempts += 1
            
            if verbose:
                if attempts%10 == 0:
                    print("Attempted fits: "+str(attempts)+", successes: "+str(successes))
            
            if sum(data.TotalCases>0)<=7:
                continue
            
            data = data.sort_values(by=['Date'], ascending = True)
            self.popdata.loc[self.popdata.Code==tc, 'LatestTotalCases'] = data.TotalCases.values[-1]
            
            model.load(data, pop, target)
            
            try:
                
                if bootstrap:
                    model.fit_bootstrap(S=25)
                    self.popdata.loc[self.popdata.Code==tc, parnames] = model.params.values()
                    self.popdata.loc[self.popdata.Code==tc, parnames_se] = model.params_sd.values()
                else:
                    model.fit()
                    self.popdata.loc[self.popdata.Code==tc, model.params.keys()] = model.params.values()
                
                successes += 1
        
            except BaseException as e:
                print("Fit failed for "+target+"due to the following error:\n"+str(e))
                continue
        
        if dropna:
        
            self.popdata.dropna(inplace=True)
        
        self.all_fitted = True
        if bootstrap:
            self.all_fitted_bootstrap = True
        