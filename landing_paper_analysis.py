import saccade_analysis as sac
import floris_misc
import numpy as np

########################################################################################################
# Collect saccade data
########################################################################################################

def collect_saccade_data(dataset, behavior=['landing', 'flyby'], last_saccade_only=True, min_angle=0):
    if type(behavior) is not list:
        behavior = [behavior]
    
    saccades = floris_misc.Object()
    saccades.keys = []
    saccades.angle_to_post = []
    saccades.angle_to_post_after_turn = []
    saccades.turn_angle = []
    saccades.true_turn_angle = []
    saccades.angle_subtended_by_post = []
    saccades.distance_to_post = []
    saccades.speed = []
    saccades.expansion = []
    saccades.angular_velocity = []
    
    if last_saccade_only:
        for k, trajec in dataset.trajecs.iteritems():
            if trajec.behavior in behavior:
                if len(trajec.last_saccade_range) > 0: 
                    sac_range = trajec.last_saccade_range
                    angle_subtended = trajec.angle_subtended_by_post[sac_range[0]]
                    if angle_subtended > min_angle*np.pi/180.:
                        angle_of_saccade = sac.get_angle_of_saccade(trajec, sac_range)
                        saccades.keys.append(k)
                        saccades.angle_to_post.append(trajec.angle_to_post[sac_range[0]])
                        saccades.angle_to_post_after_turn.append(trajec.angle_to_post[sac_range[-1]])
                        
                        saccades.true_turn_angle.append(angle_of_saccade)
                        
                        saccades.turn_angle.append(trajec.angle_to_post[sac_range[0]] - trajec.angle_to_post[sac_range[-1]])
                        
                        saccades.angle_subtended_by_post.append(trajec.angle_subtended_by_post[sac_range[0]])
                        saccades.distance_to_post.append(trajec.dist_to_stim_r_normed[sac_range[0]])
                        saccades.speed.append(trajec.speed[sac_range[0]])
                        saccades.expansion.append(trajec.expansion[sac_range[0]])
                        
                        saccades.angular_velocity.append( saccades.true_turn_angle[-1] / (float(len(sac_range))/trajec.fps) )
                        
    else:
        print 'calculating for all saccades prior to nearest approach to post'
        for k, trajec in dataset.trajecs.iteritems():
            if trajec.behavior in behavior:
                if len(trajec.sac_ranges) > 0:
                    for sac_range in trajec.sac_ranges: 
                        if sac_range[0] < trajec.frame_nearest_to_post:
                            angle_subtended = trajec.angle_subtended_by_post[sac_range[0]]
                            if angle_subtended > min_angle*np.pi/180.:
                                angle_of_saccade = sac.get_angle_of_saccade(trajec, sac_range)
                                saccades.keys.append(k)
                                saccades.angle_to_post.append(trajec.angle_to_post[sac_range[0]])
                                saccades.angle_to_post_after_turn.append(trajec.angle_to_post[sac_range[-1]])
                                
                                saccades.true_turn_angle.append(angle_of_saccade)
                                
                                saccades.turn_angle.append(trajec.angle_to_post[sac_range[0]] - trajec.angle_to_post[sac_range[-1]])
                                
                                saccades.angle_subtended_by_post.append(trajec.angle_subtended_by_post[sac_range[0]])
                                saccades.distance_to_post.append(trajec.dist_to_stim_r_normed[sac_range[0]])
                                saccades.speed.append(trajec.speed[sac_range[0]])
                                saccades.expansion.append(trajec.expansion[sac_range[0]])
                                
                                saccades.angular_velocity.append( saccades.true_turn_angle[-1] / (float(len(sac_range))/trajec.fps) )

    saccades.angle_to_post = np.array(saccades.angle_to_post)
    saccades.angle_to_post_after_turn = np.array(saccades.angle_to_post_after_turn) 
    saccades.turn_angle = np.array(saccades.turn_angle)
    saccades.true_turn_angle = np.array(saccades.true_turn_angle)
    saccades.angle_subtended_by_post = np.array(saccades.angle_subtended_by_post) 
    saccades.distance_to_post = np.array(saccades.distance_to_post) 
    saccades.speed = np.array(saccades.speed)
    saccades.expansion = np.array(saccades.expansion)
    saccades.angular_velocity = np.array(saccades.angular_velocity)
    
    saccades.angle_to_far_edge = saccades.angle_to_post + np.sign(saccades.angle_to_post)*saccades.angle_subtended_by_post/2.
    saccades.angle_to_near_edge = saccades.angle_to_post - np.sign(saccades.angle_to_post)*saccades.angle_subtended_by_post/2.
    
    dataset.saccades = saccades
