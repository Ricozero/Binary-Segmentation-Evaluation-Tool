# demo
import numpy as np
import os
from measures import compute_ave_MAE_of_methods, compute_PRE_REC_FM_of_methods, plot_save_pr_curves, plot_save_fm_curves

mode = 'calculation'
mode = 'plot'

pr_xrange = (0.5, 1.0)
pr_yrange = (0.5, 1.0)
fm_xrange = (0.0, 1.0)
fm_yrange = (0.5, 1.0)

# My paper
# gt_dir = '../Saliency Maps/GT'
# sm_dir = '../Saliency Maps/Main'
# save_dir = 'sal_eval'
# methods = os.listdir(sm_dir)
# methods.remove('Ours')
# methods.insert(0, 'Ours')
# datasets = ['HKU-IS', 'ECSSD', 'DUTS-TE', 'DUT-OMRON', 'PASCAL-S', 'SOD']
# #lineSylClr = ['r-', 'r--', 'b-', 'b--', 'g-', 'g--', 'c-', 'c--', 'm-', 'm--', 'y-', 'y--', 'k-', 'k--']
# # For gray evaluation curve image
# lineSylClr = [
#     'solid',
#     (0, (1, 1)),                # . .
#     (0, (1, 3)),                # .  .
#     (0, (2, 1)),                # - -
#     (0, (2, 3)),                # -  -
#     (0, (5, 1)),                # -- --
#     (0, (10, 1)),               # --- ---
#     (0, (1, 1, 2, 1)),          # . -
#     (0, (1, 1, 5, 1)),          # . --
#     (0, (1, 1, 10, 1)),         # . ---
#     (0, (1, 1, 1, 1, 2, 1)),    # . . -
#     (0, (1, 1, 1, 1, 5, 1)),    # . . --
#     (0, (1, 1, 2, 1, 2, 1)),    # . - -
#     (0, (1, 1, 2, 1, 5, 1)),    # . - --
# ]
# linewidth = [1.5]+[1]*(len(methods)-1)

# Ja Ning's paper
# gt_dir = 'JN/GT'
# sm_dir = 'JN/sal_map'
# save_dir = 'sal_eval_JN'
# methods = os.listdir(sm_dir)
# methods.remove('Ours')
# methods.append('Ours')
# datasets = ['ECSSD', 'DUT-OMRON']
# lineSylClr = ['b-', 'b--', 'g-', 'g--', 'c-', 'c--', 'm-', 'm--', 'y-', 'y--', 'k-', 'k--', 'r--', 'r-']
# linewidth = [1]*(len(methods)-1)+[1.5]

# Ja Ning's paper 2 (Normal and Fusion)
# gt_dir = '../Saliency Maps/GT'
# sm_dir = '../Saliency Maps/JN2-Fusion'
# #sm_dir = '../Saliency Maps/JN2'
# datasets = ['PASCAL-S', 'HKU-IS', 'ECSSD', 'DUT-OMRON', 'DUTS-TE', 'SOD']
# save_dir = 'sal_eval'
# methods = os.listdir(sm_dir)
# methods.remove('Ours')
# methods.insert(0, 'Ours')
# lineSylClr = ['r-', 'b-', 'b--', 'g-', 'g--', 'c-', 'c--', 'm-', 'm--', 'y-', 'y--', 'k-', 'k--']
# linewidth = [1.5] + [1]*(len(methods)-1)

# Ja Ning's paper 2 (New and New2)
# gt_dir = '../Saliency Maps/GT'
# #sm_dir = '../Saliency Maps/JN2-New'
# sm_dir = '../Saliency Maps/JN2-New2'
# datasets = ['PASCAL-S', 'HKU-IS', 'ECSSD', 'DUT-OMRON', 'DUTS-TE', 'SOD']
# save_dir = 'sal_eval'
# methods = os.listdir(sm_dir)
# methods.remove('Ours')
# methods.insert(0, 'Ours')
# lineSylClr = ['r-', 'b-', 'b--', 'g-', 'g--', 'c-', 'c--', 'm-', 'm--', 'y-', 'y--', 'k-', 'k--', 'r--']
# linewidth = [1.5] + [1]*(len(methods)-1)

# Ja Ning's paper 2 (for graduation essay)
# gt_dir = '../Saliency Maps/GT'
# #sm_dir = '../Saliency Maps/JN2-Fusion'
# sm_dir = '../Saliency Maps/JN2'
# datasets = ['PASCAL-S', 'HKU-IS', 'ECSSD', 'DUT-OMRON', 'DUTS-TE', 'SOD']
# save_dir = 'sal_eval'
# methods = os.listdir(sm_dir)
# methods.remove('Ours')
# methods.insert(0, 'Ours')
# lineSylClr = [
#     'solid',
#     (0, (1, 1)),                # . .
#     (0, (1, 3)),                # .  .
#     (0, (2, 1)),                # - -
#     (0, (2, 3)),                # -  -
#     (0, (5, 1)),                # -- --
#     (0, (10, 1)),               # --- ---
#     (0, (1, 1, 2, 1)),          # . -
#     (0, (1, 1, 5, 1)),          # . --
#     (0, (1, 1, 10, 1)),         # . ---
#     (0, (1, 1, 1, 1, 2, 1)),    # . . -
#     (0, (1, 1, 1, 1, 5, 1)),    # . . --
#     (0, (1, 1, 2, 1, 2, 1)),    # . - -
#     (0, (1, 1, 2, 1, 5, 1)),    # . - --
# ]
# linewidth = [1.5]+[1]*(len(methods)-1)

# Ja Ning's paper 2 (Traffic)
gt_dir = '../Saliency Maps/GT'
sm_dir = '../Saliency Maps/Traffic'
datasets = ['Traffic-TE']
save_dir = 'sal_eval'
methods = os.listdir(sm_dir)
methods.remove('Ours')
methods.insert(0, 'Ours')
lineSylClr = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-', 'k-']
linewidth = [1.5] + [1]*(len(methods)-1)
pr_xrange = (0.0, 1.0)
pr_yrange = (0.0, 0.7)
fm_yrange = (0.0, 0.7)

############################## Processing ##############################

if not os.path.exists('saves'):
    os.mkdir('saves')

for dataset in datasets:
    print('>>>>>>>>>> ' + dataset + ' <<<<<<<<<<')
    gt_img_list = os.listdir(os.path.join(gt_dir, dataset))
    gt_img_list = [os.path.join(gt_dir, dataset, gt_img) for gt_img in gt_img_list]
    methods_tmp = methods.copy()
    for method in methods_tmp:
        if not os.path.exists(os.path.join(sm_dir, method, dataset)):
            methods_tmp.remove(method)
    sm_dir_list = [os.path.join(sm_dir, method) for method in methods_tmp]

    if mode == 'calculation':
        # print("------Compute the average MAE of Methods------")
        # aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_img_list,sm_dir_list)
        # for i in range(0,len(sm_dir_list)):
        #     print('>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.3f'%(sm_dir_list[i], gt2rs_mae[i], len(gt_img_list), aveMAE[i]))
        # print('\n')

        print("------Compute the Precision, Recall and F-measure of Methods------")
        PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_img_list,sm_dir_list,dataset,beta=0.3)
        for i in range(0,FM.shape[0]):
            print(">>", sm_dir_list[i],":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_img_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
        print('\n')
        np.save(os.path.join('saves', dataset + '_PRE.npy'), PRE)
        np.save(os.path.join('saves', dataset + '_REC.npy'), REC)
        np.save(os.path.join('saves', dataset + '_FM.npy'), FM)

        # Write result log
        # with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
        #     f.write(dataset + '\n')
        #     f.write('method MAE maxF meanF' + '\n')
        #     for i in range(0,FM.shape[0]):
        #         f.write('%s\t%.3f %.3f %.3f\n' % (os.path.basename(sm_dir_list[i]), aveMAE[i], np.max(FM,1)[i], np.mean(FM,1)[i]))
        #     f.write('\n')
    elif mode == 'plot':
        PRE = np.load(os.path.join('saves', dataset + '_PRE.npy'))
        REC = np.load(os.path.join('saves', dataset + '_REC.npy'))
        FM = np.load(os.path.join('saves', dataset + '_FM.npy'))

    ## =======Plot and save precision-recall curves=========
    print("------Plot and save precision-recall curves------")
    plot_save_pr_curves(PRE, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        REC, # numpy array (num_rs_dir,255)
                        method_names = sm_dir_list, # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
                        linewidth = linewidth, # curve width, shape (num_rs_dir)
                        xrange = pr_xrange, # the showing range of x-axis
                        yrange = pr_yrange, # the showing range of y-axis
                        dataset_name = dataset, # dataset name will be drawn on the bottom center position
                        save_dir = save_dir, # figure save directory
                        save_fmt = 'png') # format of the to-be-saved figure
    print('\n')

    ## =======Plot and save F-measure curves=========
    print("------Plot and save F-measure curves------")
    plot_save_fm_curves(FM, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
                        mybins = np.arange(0,256),
                        method_names = sm_dir_list, # method names, shape (num_rs_dir), will be included in the figure legend
                        lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
                        linewidth = linewidth, # curve width, shape (num_rs_dir)
                        xrange = fm_xrange, # the showing range of x-axis
                        yrange = fm_yrange, # the showing range of y-axis
                        dataset_name = dataset, # dataset name will be drawn on the bottom center position
                        save_dir = save_dir, # figure save directory
                        save_fmt = 'png') # format of the to-be-saved figure
    print('\n')

print('Done!!!')
