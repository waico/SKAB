import pandas as pd

def evaluating_change_point(true, prediction, metric='nab', numenta_time=None):
    """
    true - both:
                list of pandas Series with binary int labels
                pandas Series with binary int labels
    prediction - both:
                      list of pandas Series with binary int labels
                      pandas Series with binary int labels
    metric: 'nab', 'binary' (FAR, MAR), 'average_delay'
                
    """
    
    def binary(true, prediction):      
        """
        true - true binary series with 1 as anomalies
        prediction - trupredicted binary series with 1 as anomalies
        """
        def single_binary(true,prediction):
            true_ = true == 1 
            prediction_ = prediction == 1
            TP = (true_ & prediction_).sum()
            TN = (~true_ & ~prediction_).sum()
            FP = (~true_ & prediction_).sum()
            FN = (true_ & ~prediction_).sum()
            return TP,TN,FP,FN
            
        if type(true) != type(list()):
            TP,TN,FP,FN = single_binary(true,prediction)
        else:
            TP,TN,FP,FN = 0,0,0,0
            for i in range(len(true)):
                TP_,TN_,FP_,FN_ = single_binary(true[i],prediction[i])
                TP,TN,FP,FN = TP+TP_,TN+TN_,FP+FP_,FN+FN_       
    
        f1 = round(TP/(TP+(FN+FP)/2), 2)
        print(f'False Alarm Rate {round(FP/(FP+TN)*100,2)} %' )
        print(f'Missing Alarm Rate {round(FN/(FN+TP)*100,2)} %')
        print(f'F1 metric {f1}')
        return f1
    
    def average_delay(detecting_boundaries, prediction):
        
        def single_average_delay(detecting_boundaries, prediction):
            missing = 0
            detectHistory = []
            for couple in detecting_boundaries:
                t1 = couple[0]
                t2 = couple[1]
                if prediction[t1:t2].sum()==0:
                    missing+=1
                else:
                    detectHistory.append(prediction[prediction ==1][t1:t2].index[0]-t1)
            return missing, detectHistory
            
        
        if type(prediction) != type(list()):
            missing, detectHistory = single_average_delay(detecting_boundaries, prediction)
        else:
            missing, detectHistory = 0, []
            for i in range(len(prediction)):
                missing_, detectHistory_ = single_average_delay(detecting_boundaries[i], prediction[i])
                missing, detectHistory = missing+missing_, detectHistory+detectHistory_

        add = pd.Series(detectHistory).mean()
        print('Average delay', add)
        print(f'A number of missed CPs = {missing}')
        return add
    
    def evaluate_nab(detecting_boundaries, prediction, table_of_coef=None):
        print(detecting_boundaries)
        """
        Scoring labeled time series by means of
        Numenta Anomaly Benchmark methodics
        Parameters
        ----------
        detecting_boundaries: list of list of two float values
            The list of lists of left and right boundary indices
            for scoring results of labeling
        prediction: pd.Series with timestamp indices, in which 1 
            is change point, and 0 in other case. 
        table_of_coef: pandas array (3x4) of float values
            Table of coefficients for NAB score function
            indeces: 'Standart','LowFP','LowFN'
            columns:'A_tp','A_fp','A_tn','A_fn'
        Returns
        -------
        Scores: numpy array, shape of 3, float
            Score for 'Standart','LowFP','LowFN' profile 
        Scores_null: numpy array, shape 3, float
            Null score for 'Standart','LowFP','LowFN' profile             
        Scores_perfect: numpy array, shape 3, float
            Perfect Score for 'Standart','LowFP','LowFN' profile  
        """
        def single_evaluate_nab(detecting_boundaries,
                                prediction, 
                                table_of_coef=None,
                                name_of_dataset=None,
                                intersection_mode='cut left',
                                mode_cp_or_ad = 'cp',
                                scale_func = "default",
                                scale_koef=2):
            def sigm_scale(y, A_tp, A_fp, window=1):
                    return (A_tp-A_fp)*(1/(1+np.exp(5*y/window))) + A_fp
            def my_scale(len_ts,A_tp,A_fp,koef=1):
                """ts - участок на котором надо жахнуть окно """
                x = np.linspace(-np.pi/2,np.pi/2,len_ts)
                # Приведение если неравномерный шаг.
                #x_new = x_old * ( np.pi / (x_old[-1]-x_old[0])) - x_old[0]*( np.pi / (x_old[-1]-x_old[0])) - np.pi/2
                y = (A_tp-A_fp)/2*-1*np.tanh(koef*x)/(np.tanh(np.pi*koef/2)) + (A_tp-A_fp)/2 + A_fp
                return y 

            if scale_func == "default":
                scale_func = my_scale            
            
            if table_of_coef is None:
                table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                     [1.0,-0.22,1.0,-1.0],
                                      [1.0,-0.11,1.0,-2.0]])
                table_of_coef.index = ['Standart','LowFP','LowFN']
                table_of_coef.index.name = "Metric"
                table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']

            detecting_boundaries = detecting_boundaries.copy()
            prediction = prediction.copy()
            _df_fill_bounds =  pd.DataFrame(np.ones((len(prediction),len(detecting_boundaries)))*np.nan,index=prediction.index)
            for i in range(len(detecting_boundaries)):
                _df_fill_bounds.loc[detecting_boundaries[i][0]:detecting_boundaries[i][1],i]=1
            
            Scores, Scores_perfect, Scores_null=[], [], []
            for profile in ['Standart', 'LowFP', 'LowFN']:       
                A_tp = table_of_coef['A_tp'][profile]
                A_fp = table_of_coef['A_fp'][profile]
                A_fn = table_of_coef['A_fn'][profile]
                
                score = 0
                # FPs
                ts_fp = pd.Series(np.ones(len(prediction)),index=prediction.index)
                ts_fp.loc[_df_fill_bounds.dropna(how='all').index]=0
                ts_fp = ts_fp * prediction
                score += A_fp*ts_fp.sum()
                #FNs and TPs
                for i in range(len(detecting_boundaries)):
                    ts_tp = _df_fill_bounds.iloc[:,i].dropna()
                    ts_tp = ts_tp * prediction.loc[ts_tp.index]
                    if ts_tp.sum()==0:
                        score+=A_fn
                        
                    else:
                        ts_profile = pd.Series(data=scale_func(
                                                                len(ts_tp),A_tp,A_fp,koef=scale_koef),
                                               index = ts_tp.index)
                        plt.plot(ts_profile)
                        ts_tp.loc[ts_tp[ts_tp==1].index[1:]] = 0 
                        plt.axvline(ts_tp[ts_tp==1].index[0])
                        score = (ts_profile * ts_tp).sum()
                print(score)
                Scores.append(score)
                Scores_perfect.append(len(detecting_boundaries)*A_tp)
                Scores_null.append(len(detecting_boundaries)*A_fn)
            return np.array([np.array(Scores),np.array(Scores_null), np.array(Scores_perfect)])
       #======      
        if type(prediction) != type(list()):
            matrix = single_evaluate_nab(detecting_boundaries, prediction, table_of_coef=table_of_coef)
        else:
            matrix = np.zeros((3,3))
            for i in range(len(prediction)):
                matrix_ = single_evaluate_nab(detecting_boundaries[i], prediction[i], table_of_coef=table_of_coef,name_of_dataset=i)
                matrix = matrix + matrix_      
                
        results = {}
        desc = ['Standart', 'LowFP', 'LowFN'] 
        for t, profile_name in enumerate(desc):
            results[profile_name] = round(100*(matrix[0,t]-matrix[1,t])/(matrix[2,t]-matrix[1,t]), 2)
            print(profile_name,' - ', results[profile_name])
        
        return results
            
            
    #=========================================================================
    if type(true) != type(list()):
        true_items = true[true==1].index
    else:
        true_items = [true[i][true[i]==1].index for i in range(len(true))]
        

    if not metric=='binary':
        def single_detecting_boundaries(true, numenta_time, true_items):
            detecting_boundaries=[]
            td = pd.Timedelta(numenta_time) if numenta_time is not None else pd.Timedelta((true.index[-1]-true.index[0])/len(true_items))  
            for val in true_items:
                detecting_boundaries.append([val, val + td])
            return detecting_boundaries
        
        if type(true) != type(list()):
            detecting_boundaries = single_detecting_boundaries(true=true, numenta_time=numenta_time, true_items=true_items)
        else:
            detecting_boundaries=[]
            for i in range(len(true)):
                detecting_boundaries.append(single_detecting_boundaries(true=true[i], numenta_time=numenta_time, true_items=true_items[i]))
        # block for resolving intersection problem:
        # важно не ошибиться, и всегда следить, чтобы везде правая граница далее
        # не включалась, иначе будет пересечения окон             
        new_detecting_boundaries = detecting_boundaries.copy() 
        if new_detecting_boundaries[0][0] < prediction.index[0]:
            new_detecting_boundaries[0][0] = prediction.index[0]
        if new_detecting_boundaries[-1][-1] > prediction.index[-1]:
            new_detecting_boundaries[-1][-1] = prediction.index[-1]
        for i in range(len(new_detecting_boundaries)-1):
            if new_detecting_boundaries[i][1] >= new_detecting_boundaries[i+1][0]:
                print("Intersection of scoring windows")
                if intersection_mode == 'cut left':
                    new_detecting_boundaries[i][1] = new_detecting_boundaries[i+1][0]
                elif intersection_mode == 'cut right':
                    new_detecting_boundaries[i+1][0] = new_detecting_boundaries[i][1]
                elif intersection_mode == 'cut both':
                    _a  = new_detecting_boundaries[i][1]
                    new_detecting_boundaries[i][1] = new_detecting_boundaries[i+1][0]
                    new_detecting_boundaries[i+1][0] = _a
                else:
                    raise("choose the intersection_mode")
        detecting_boundaries = new_detecting_boundaries.copy()

    if metric== 'nab':
        return evaluate_nab(detecting_boundaries, prediction)
    elif metric=='average_delay':
        return average_delay(detecting_boundaries, prediction)
    elif metric== 'binary':
        return binary(true, prediction)