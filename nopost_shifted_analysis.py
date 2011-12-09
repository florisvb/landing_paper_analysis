import flydra_floris_analysis as ffa


import flydra_analysis_dataset as fad

import flydra_analysis as fa

import floris_math as fm
import saccade_analysis

import copy
import numpy as np

import trajectory_analysis_specific as tas

def shift_no_post_dataset(dataset, prefix='1_', shift_range=0.06):
    dataset_shifted = ffa.Dataset(like=dataset)
    for k, trajec in dataset.trajecs.iteritems():
        new_k = prefix + k
        new_trajec = copy.deepcopy(trajec)
        new_trajec.key = new_k
        new_trajec.positions[:, 0] += (np.random.random()*shift_range*2.+0.03*2) - (shift_range+0.03)
        dataset_shifted.trajecs.setdefault(new_k, new_trajec)
    return dataset_shifted
    
def shift_datasets(dataset, nshifts=2):
    shifted_dataset_list = []
    prefix = 0
    for i in range(nshifts):
        prefix += 1
        s = str(prefix) + '_'
        shifted_dataset = shift_no_post_dataset(dataset, prefix=s, shift_range=0.05)
        shifted_dataset_list.append(copy.deepcopy(shifted_dataset))

    #dataset_merged = fa.add_datasets(shifted_dataset_list)
    #fad.save(dataset_merged, 'shifted_dataset')
    return shifted_dataset_list

def classify_false_post(trajec):
    trajec.calc_dist_to_stim()
    trajec.calc_dist_to_stim_r()
    if np.min(trajec.dist_to_stim_r) < 0.005:
        trajec.behavior = 'landing'
    else:
        trajec.behavior = 'flyby'
        
def calc_frame_of_landing(trajec):    
    if trajec.behavior == 'landing':
        trajec.frame_of_landing = np.argmin(trajec.dist_to_stim_r)
    
    elif trajec.behavior == 'flyby':
        trajec.frame_of_landing = trajec.length
        
        
        

    
    
    
def make_behavior_dataset(dataset, filename='dataset_nopost_landing', behavior='landing'):

    REQUIRED_LENGTH = 30
    REQUIRED_DIST = 0.1

    new_dataset = ffa.Dataset(like=dataset)
    if type(behavior) is not list:
        behavior = [behavior]
    for k,trajec in dataset.trajecs.items():
        trajec.key = k
        if trajec.behavior in behavior:
            
            calc_frame_of_landing(trajec)
            fa.normalize_dist_to_stim_r(trajec)
            saccade_analysis.calc_saccades2(trajec)
            
            if trajec.behavior == 'landing':
                if trajec.dist_to_stim_r_normed[0] >= REQUIRED_DIST:
                    #if np.max(trajec.positions[:,2]) < 0 and np.min(trajec.positions[:,2]) > -0.15:
                    if np.min(trajec.positions[:,2]) > -0.15:
                        trajec.frames = np.arange(fa.get_frame_at_distance(trajec, REQUIRED_DIST), trajec.frame_of_landing).tolist()
                        if trajec.frame_of_landing > REQUIRED_LENGTH:
                            if np.max(trajec.positions[trajec.frames,2]) < 0.15: # no real altitude check
                                #classify(trajec, dfar=REQUIRED_DIST, dnear=0.005)
                                new_dataset.trajecs.setdefault(k, trajec)
                                
                                first_frame = 0
                                trajec.frames_below_post = np.arange(first_frame, trajec.frame_of_landing+1).tolist()

            elif trajec.behavior == 'flyby':
                frame_nearest_to_post = np.argmin(trajec.dist_to_stim_r)
                print k
                if frame_nearest_to_post > 10 and np.max(trajec.dist_to_stim_r[0:frame_nearest_to_post]) > REQUIRED_DIST:
                    if np.min(trajec.positions[:,2]) > -0.15:
                        if trajec.dist_to_stim_r[frame_nearest_to_post] < 0.1:
                            fs = np.arange(frame_nearest_to_post,len(trajec.speed)).tolist()
                            try:
                                last_frame = get_frame_at_distance(trajec, REQUIRED_DIST, frames=fs)
                            except:
                                last_frame = len(trajec.speed)-1
                            first_frame = fa.get_frame_at_distance(trajec, REQUIRED_DIST, frames=np.arange(0,frame_nearest_to_post).tolist())
                            
                            trajec.frames = np.arange(first_frame, last_frame).tolist()
                            
                            # get frame at 8cm away, prior to nearest approach
                            frame_nearest_to_post = np.argmin(trajec.dist_to_stim_r)
                            trajec.frame_nearest_to_post = frame_nearest_to_post
                            frames = np.arange(0, frame_nearest_to_post).tolist()
                            trajec.frames_of_flyby = frames
                            frame_at_distance = fa.get_frame_at_distance(trajec, 0.08, singleframe=True, frames=frames)
                            
                            last_frame = np.min( [frame_nearest_to_post+20, len(trajec.speed)-1]) 
                            
                            sacs = [s[0] for s in trajec.sac_ranges]
                            sac_sgns = np.array(sacs) - frame_at_distance
                            sac_negs = np.where(sac_sgns<0)[0]
                            if len(sac_negs) > 0:
                                sac_neg = sac_negs[0]
                            else:
                                sac_neg = 0
                            first_frame = sac_neg + 1
                            
                            try:
                                trajec.frames_of_flyby = np.arange(first_frame, last_frame).tolist()
                                new_dataset.trajecs.setdefault(k, trajec)
                            except:
                                print 'ignored key: ', k, first_frame, last_frame
                            
                            new_dataset.trajecs.setdefault(k, trajec)
                            
            
    fa.save(new_dataset, filename)

    return new_dataset
    
    
    

def make_false_landing_and_flyby_datasets(dataset_nopost):
    
    #shift_list = shift_datasets(dataset_nopost, nshifts=nshifts)
    #shift = fa.add_datasets(shift_list)
    
    shift = dataset_nopost
    
    fa.calc_func(shift, classify_false_post)
    fa.calc_func(shift, calc_frame_of_landing)
    
    dataset_nopost_landing = make_behavior_dataset(shift, filename='dataset_nopost_landing', behavior='landing')
    
    dataset_nopost_flyby = make_behavior_dataset(shift, filename='dataset_nopost_flyby', behavior='flyby')
    
    fa.prep_dataset(dataset_nopost_landing)
    fa.calc_func(dataset_nopost_landing, saccade_analysis.calc_last_saccade)
    
    fa.prep_dataset(dataset_nopost_flyby)
    fa.calc_func(dataset_nopost_flyby, saccade_analysis.calc_last_saccade)
    
    #fa.save(dataset_nopost_landing, 'dataset_nopost_landing')
    fa.save(dataset_nopost_flyby, 'dataset_nopost_flyby')
    
    return dataset_nopost_landing, dataset_nopost_flyby
    
    
    
    
    
    
    
