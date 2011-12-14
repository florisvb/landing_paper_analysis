from rpy2 import robjects
import numpy as np

import generate_paper_figs as gpf

def linear_model_of_landing_decelerations(dataset):
    
    speeds = []
    angles_subtended = []
    
    speeds_s = []
    angles_subtended_s = []
    
    for k, trajec in dataset.trajecs.items():
        if trajec.frame_at_deceleration is not None:
            fd = trajec.frame_at_deceleration
            if len(trajec.last_saccade_range) > 0:
                if fd > trajec.last_saccade_range[-1]:
                    speeds.append(trajec.speed[fd])
                    angles_subtended.append(trajec.angle_subtended_by_post[fd])
                else:
                    speeds_s.append(trajec.speed[fd])
                    angles_subtended_s.append(trajec.angle_subtended_by_post[fd])
            else:
                speeds.append(trajec.speed[fd])
                angles_subtended.append(trajec.angle_subtended_by_post[fd])
                
    angles_subtended = np.log(np.array(angles_subtended)).tolist()
    angles_subtended_s = np.log(np.array(angles_subtended_s)).tolist()
           
    # compile lists 
    ylist = speeds + speeds_s
    xlist = angles_subtended + angles_subtended_s
    x2list = [0 for i in range(len(speeds))] + [1 for i in range(len(speeds_s))]

    r = robjects.r

    x = robjects.FloatVector(xlist)
    y = robjects.FloatVector(ylist)
    x2 = robjects.FloatVector(x2list)

    robjects.globalEnv["y"] = y
    robjects.globalEnv["x"] = x
    robjects.globalEnv["x2"] = x2
    lm = r.lm("y~x+x2+x:x2")
    print r.summary(lm)
    #print r.aov(lm)




def linear_model_of_landing_post_type(dataset, do_r=False):
    
    speeds_black = []
    angles_subtended_black = []
    
    speeds_checkered = []
    angles_subtended_checkered = []
    
    for k, trajec in dataset.trajecs.items():
        if trajec.frame_at_deceleration is not None:
            fd = trajec.frame_at_deceleration
            if 'checkered' in trajec.post_type:
                speeds_checkered.append(trajec.speed[fd])
                angles_subtended_checkered.append(trajec.angle_subtended_by_post[fd])
            elif 'black' in trajec.post_type:
                speeds_black.append(trajec.speed[fd])
                angles_subtended_black.append(trajec.angle_subtended_by_post[fd])
   
    angles_subtended_black = np.log(np.array(angles_subtended_black)).tolist()
    angles_subtended_checkered = np.log(np.array(angles_subtended_checkered)).tolist()  
    
    print 'n black: ', len(angles_subtended_black)
    print 'n checkered: ', len(angles_subtended_checkered)
                           
    # compile lists 
    if do_r:
        ylist = speeds_black + speeds_checkered
        xlist = angles_subtended_black + angles_subtended_checkered
        x2list = [0 for i in range(len(speeds_black))] + [1 for i in range(len(speeds_checkered))]

        r = robjects.r

        x = robjects.FloatVector(xlist)
        y = robjects.FloatVector(ylist)
        x2 = robjects.FloatVector(x2list)

        robjects.globalEnv["y"] = y
        robjects.globalEnv["x"] = x
        robjects.globalEnv["x2"] = x2
        lm = r.lm("y~x+x2+x:x2")
        print r.summary(lm)
        #print r.aov(lm)
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    
    
    yticks = np.linspace(0,1,5,endpoint=True)
    gpf.set_log_ticks(ax, yticks=yticks, radians=True)
    ax.set_ylabel('Speed, m/s')
    
    print 'range of angles to post: '
    print 'min: ', np.min(angles_to_post)
    print 'max: ', np.max(angles_to_post)
    print 'mean: ', np.mean(angles_to_post)
    print 'std: ', np.std(angles_to_post)
                
    fig_width = 3.25 # width in inches
    fig_height = 3.25  # height in inches
    fig_size =  (fig_width, fig_height)
    fig.set_size_inches(fig_size)
    fig.savefig('landing_deceleration_plot.pdf', format='pdf')
    '''
    
    
    
    
def linear_model_of_flyby_post_type(dataset, do_r=False):
    
    turnangle_black = []
    postangle_black = []
    size_black = []
    
    turnangle_checkered = []
    postangle_checkered = []
    size_checkered = []
    
    saccades = dataset.saccades
    
    '''
    right = np.where( (turn_angle < 0)*(turn_angle < angle_to_post) )[0].tolist()
    right_outliers = np.where( (turn_angle < 0)*(turn_angle > angle_to_post) )[0].tolist()
    left = np.where( (turn_angle > 0)*(turn_angle > angle_to_post) )[0].tolist()
    left_outliers = np.where( (turn_angle > 0)*(turn_angle < angle_to_post) )[0].tolist()
    '''
    
    for i in range(len(saccades.true_turn_angle)):
        
        if (saccades.true_turn_angle[i] < 0)*(saccades.true_turn_angle[i] < saccades.angle_to_post[i]):
            
            if 1 == saccades.post_type[i]:
                turnangle_black.append(saccades.true_turn_angle[i])
                postangle_black.append(saccades.angle_to_post[i])
                size_black.append(saccades.angle_subtended_by_post[i])
            if 0 == saccades.post_type[i]:
                turnangle_checkered.append(saccades.true_turn_angle[i])
                postangle_checkered.append(saccades.angle_to_post[i])
                size_checkered.append(saccades.angle_subtended_by_post[i])
                
    print 'n black: ', len(turnangle_black)
    print 'n checkered: ', len(turnangle_checkered)
                           
    # compile lists 
    if do_r:
        ylist = turnangle_black + turnangle_checkered
        xlist = postangle_black + postangle_checkered
        #x2list = [0 for i in range(len(turnangle_black))] + [1 for i in range(len(turnangle_checkered))]
        x2list = size_black + size_checkered
        
        r = robjects.r

        x = robjects.FloatVector(xlist)
        y = robjects.FloatVector(ylist)
        x2 = robjects.FloatVector(x2list)

        robjects.globalEnv["y"] = y
        robjects.globalEnv["x"] = x
        robjects.globalEnv["x2"] = x2
        lm = r.lm("y~x+x2+x:x2")
        print r.summary(lm)
        #print r.aov(lm)
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    
    
    
    yticks = np.linspace(0,1,5,endpoint=True)
    gpf.set_log_ticks(ax, yticks=yticks, radians=True)
    ax.set_ylabel('Speed, m/s')
    
    print 'range of angles to post: '
    print 'min: ', np.min(angles_to_post)
    print 'max: ', np.max(angles_to_post)
    print 'mean: ', np.mean(angles_to_post)
    print 'std: ', np.std(angles_to_post)
                
    fig_width = 3.25 # width in inches
    fig_height = 3.25  # height in inches
    fig_size =  (fig_width, fig_height)
    fig.set_size_inches(fig_size)
    fig.savefig('landing_deceleration_plot.pdf', format='pdf')
    '''
    
    
