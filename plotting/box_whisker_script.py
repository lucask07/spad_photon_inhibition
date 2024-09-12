import matplotlib as mpl
mpl.rcParams['lines.markersize']=4

plt.ion()

# fig,ax = plt.subplots(2,2,figsize=(5,3.5))
fig,ax = plt.subplots(2,2,figsize=(3.25, 2.275))

for pp_idx, pp in enumerate(pps):
    print(f'PP / bracket {pp}')
    print('-'*40)
    eq_metric = MetricEqual(metric='ssim', vals=[0.7, 0.75, 0.8])

    # for each row in the data frame load the inhibition result 
    for ridx,row in df.iterrows():
        #print(f'At row {ridx}')

        irs = load_irs(row=row)
        if pp == 'bracket':
            vs = eq_metric.get_equals(irs.metrics)
            
        elif pp==1:
            ir = irs.find_ppp(pp)[0]
            vs = eq_metric.get_equals(ir.metrics)

        for v in vs:
            df.loc[ridx, v['cols']] = v['vals'] 

    ils = np.unique(df['inhibit_lens'])
    its = np.unique(df['inhibit_threshs'])
    kernels = np.unique(df['kernels'])

    for il in ils:
        for it in its:
            for k in kernels:
                idx = (df.inhibit_lens==il) & (df.inhibit_threshs==it) & (df.kernels==k) & (df['mask']=='nomask')
                print(np.sum(idx)) # this is the number of unique images (19 in total) 
                dfs = df[idx] # get mask values (.mask doesn't work) 
                for det_or_meas in ['det', 'meas']:
                    for val_idx, val in enumerate(eq_metric.vals):
                        tmp = np.average(dfs[f'{eq_metric.metric}_{val_idx}_{det_or_meas}_savings'])
                        df.loc[idx, f'{eq_metric.metric}_{val_idx}_{det_or_meas}_savings_avg'] = tmp # this is the average over all images and is replicated into the row of each image  
    det_or_meas = 'det'
    df_j = pd.DataFrame()
    
    for ax_col, val_idx in enumerate([0,2]):

        # print the best performing policies based on power savings 
        
        if ax_col == 0:
            df_sort = df.sort_values(f'{eq_metric.metric}_{val_idx}_{det_or_meas}_savings_avg')
            sorted_policies = [] # only determine the optimal policies for the first value of a given exposure level
        for k in kernels:
            print(k)
            # .iloc[0] selects the first image 
            print(df_sort[df_sort.kernels==k].iloc[0][f'ssim_{val_idx}_det_savings_avg'])
            print(df_sort[df_sort.kernels==k].iloc[0])
            if ax_col==0:
                sorted_policies.append({'il': df_sort[df_sort.kernels==k].iloc[0]['inhibit_lens'], 'it': df_sort[df_sort.kernels==k].iloc[0]['inhibit_threshs'], 'k': k})
                tmp = df_sort[ (df_sort.kernels==k) & (df_sort.inhibit_lens==sorted_policies[-1]['il']) & (df_sort.inhibit_threshs==sorted_policies[-1]['it']) & (df_sort['mask']=='nomask')]
                df_j = pd.concat([df_j, tmp])

        # create a jitter plot for 
        # select a subset of the dataframe 
        k_p = ['flip_laplacian', 'laplacian', 'neighbor8', 'single_pix_bright']
        
        sns.boxplot(x='kernels', y=f'ssim_{val_idx}_det_savings', data=df_j[df_j.kernels.isin(k_p)], boxprops=dict(alpha=0.2), ax=ax[pp_idx, ax_col], fliersize=0)
        sns.stripplot(x='kernels', y=f'ssim_{val_idx}_det_savings', data=df_j[df_j.kernels.isin(k_p)], jitter=0.2, ax=ax[pp_idx, ax_col], size=2)

        if ax_col==0:
            ax[pp_idx, ax_col].set_ylabel('D [%]')
        else:
            ax[pp_idx, ax_col].set_ylabel('')
        if pp_idx==1:
            ax[pp_idx,ax_col].set_xlabel('Policy')
        else:
            ax[pp_idx,ax_col].set_xlabel('')

        if pp_idx == 0:
            ax[pp_idx,ax_col].set_ylim([-60, 40])
            ax[pp_idx,ax_col].set_yticks([-40,-20,0,20,40])
        elif pp_idx == 1:
            ax[pp_idx,ax_col].set_yticks([-20,0,20,40])
            ax[pp_idx,ax_col].set_ylim([-30, 40])

        if pp_idx == 0:
            # ax[pp_idx,ax_col].set_xticklabels(['$P_{cr}$\n(proposed)', '$P_L$', '$P_{avg}$','$P_s$'])   
            ax[pp_idx,ax_col].set_xticklabels(['$P_{cr}$', '$P_L$', '$P_{avg}$','$P_s$'])   
        elif pp_idx == 1:
            # ax[pp_idx,ax_col].set_xticklabels(["$P_{cr}'$\n(proposed)", "$P_L'$", "$P_{avg}'$","$P_s'$"])   
            ax[pp_idx,ax_col].set_xticklabels(["$P_{cr}'$", "$P_L'$", "$P_{avg}'$","$P_s'$"])   
        ax[pp_idx,ax_col].set_yticks([-40,-20,0,20])
        xl,xm = ax[pp_idx,ax_col].get_xlim()
        ax[pp_idx,ax_col].hlines(0, xl, xm, linewidth=0.5, linestyles='dashed')

ax[0,1].yaxis.set_ticklabels([])
ax[1,1].yaxis.set_ticklabels([])
# plt.figtext(0.47, 0.985, 'Exposure Brackets')
# plt.figtext(0.48, 0.51, 'Single Exposure')

fig.tight_layout(pad=1.02)
my_savefig(fig, figure_dir, 'box_whisker_multiple_image_results_fullcol')
