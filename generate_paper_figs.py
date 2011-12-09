import floris_plot_lib as fpl
import matplotlib.pyplot as plt
import numpy as np
import linear_fit
import floris_math
import fit2dpolynomial
import matplotlib
import copy


def make_colorbar():
    ticks = [0, 0.05, .1]
    fpl.colorbar(ax=None, ticks=ticks, colormap='jet', aspect=20, orientation='horizontal', filename='colorbar_for_saccade_plots.pdf', flipspine=True)
    

def saccade_plot_landings_and_flybys(dataset_landing, dataset_flyby):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if 0:
        ax.scatter(dataset_flyby.saccades.angle_to_post*180/np.pi, dataset_flyby.saccades.true_turn_angle*180./np.pi/2, c='red', linewidth=0, s=3)
        ax.scatter(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.true_turn_angle*180./np.pi/2, c='green', linewidth=0, s=3)
    
    if 1:
        ax.scatter(dataset_flyby.saccades.angle_to_post*180/np.pi, dataset_flyby.saccades.angle_to_post*180/np.pi-dataset_flyby.saccades.true_turn_angle*180./np.pi, c='red', linewidth=0, s=3)
        ax.scatter(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.angle_to_post*180/np.pi-dataset_landing.saccades.true_turn_angle*180./np.pi, c='green', linewidth=0, s=3)
    
    fig.savefig('saccades_plot_landing_and_flyby.pdf', format='pdf')
    
    
def angle_to_post_before_saccade(dataset_flyby, dataset_nopost_flyby):

    flyby_angle_before = dataset_flyby.saccades.angle_to_post*180./np.pi
    nopost_angle_before = dataset_nopost_flyby.saccades.angle_to_post*180./np.pi
    
    data_all = np.hstack([flyby_angle_before, nopost_angle_before])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    bins = np.linspace(-540, 540, 80, endpoint=True)
    fpl.histogram(ax, [flyby_angle_before, nopost_angle_before], colors=['black', 'green'], bins=bins, edgecolor='none', bar_alpha=0.8, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    
    # prep plot    
    def prep_plot(ax, sym=True):
        xticks = [-180, -90, 0, 90, 180]
        yticks = [0, .01]
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks, smart_bounds=True)
        ax.set_aspect('auto')
        
        ax.set_xlim([np.min(xticks), np.max(xticks)])
        ax.set_ylim([np.min(yticks), np.max(yticks)])
        
        ax.set_xlabel('Angle to post before turn')
        ax.set_ylabel('Number of saccades')
        
    prep_plot(ax)
    
    fig.savefig('angle_to_post_before_turn.pdf')
    
    
    
def angle_to_post_after_saccade(dataset_flyby, dataset_landing, dataset_nopost_flyby, dataset_nopost_landing, nopost=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    figorig = plt.figure()
    axorig = figorig.add_subplot(111)
    
    figbefore = plt.figure()
    axbefore = figbefore.add_subplot(111)
    
    figafter = plt.figure()
    axafter = figafter.add_subplot(111)
    
    if nopost is False:
        bins = np.linspace(-540, 540, 80, endpoint=True)
    else:
        bins = np.linspace(-540, 540, 50, endpoint=True)
    if 0:
        flyby_angle_after = (dataset_flyby.saccades.angle_to_post-dataset_flyby.saccades.true_turn_angle)*180/np.pi
        landing_angle_after = (dataset_landing.saccades.angle_to_post-dataset_landing.saccades.true_turn_angle)*180/np.pi
    else:
        flyby_angle_after = dataset_flyby.saccades.angle_to_post_after_turn*180./np.pi
        landing_angle_after = dataset_landing.saccades.angle_to_post_after_turn*180./np.pi
        
    # before
    flyby_angle_before = dataset_flyby.saccades.angle_to_post*180./np.pi
    landing_angle_before = dataset_landing.saccades.angle_to_post*180./np.pi
    data_all_before = np.hstack([flyby_angle_before, landing_angle_before])
        
    nopost_flyby_angle_before = dataset_nopost_flyby.saccades.angle_to_post*180./np.pi
    nopost_landing_angle_before = dataset_nopost_landing.saccades.angle_to_post*180./np.pi
    nopost_data_all_before = np.hstack([nopost_flyby_angle_before, nopost_landing_angle_before])
        
    # no post after
    nopost_flyby_angle_after = dataset_nopost_flyby.saccades.angle_to_post_after_turn*180./np.pi
    nopost_landing_angle_after = dataset_nopost_landing.saccades.angle_to_post_after_turn*180./np.pi
    nopost_data_all = np.hstack([nopost_flyby_angle_after, nopost_landing_angle_after])    
        
    data_all = np.hstack([flyby_angle_after, landing_angle_after])
    data_flyby_symm = np.hstack([flyby_angle_after, flyby_angle_after*-1])
    data_landing_symm = np.hstack([landing_angle_after, landing_angle_after*-1])
    
    data_all_symm = np.hstack([data_all, data_all*-1])
    data_all_symm_wrapped = np.hstack([data_all_symm, data_all_symm+360, data_all_symm-360])
    
    fpl.histogram(axbefore, [data_all_before, nopost_data_all_before], colors=['black', 'green'], bins=bins, edgecolor='none', bar_alpha=0.8, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    fpl.histogram(axafter, [data_all, nopost_data_all], colors=['black', 'green'], bins=bins, edgecolor='none', bar_alpha=0.8, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    
    
    fpl.histogram(ax, [data_all_symm], colors=['black'], bins=bins, edgecolor='none', bar_alpha=0.8, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=False, normed=False, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.5)
    fpl.histogram(ax2, [data_flyby_symm, data_landing_symm], colors=['orange', 'blue'], bins=bins, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=False, normed=False, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    
    xdata = np.linspace(-180, 180, 100, endpoint=True)
    
    flyby_mean = 140
    flyby_std = 50
    landing_mean = 0
    landing_std = 40
    
    # flyby pos gaussian:
    gaus_flyby_pos = np.exp(-1*(xdata-flyby_mean)**2 / (2*flyby_std**2)) * (1/(flyby_std*np.sqrt(2*np.pi)))
    #gaus_flyby_pos_integral = np.sum(gaus_flyby_pos*floris_math.diffa(xdata))
    #print gaus_flyby_pos_integral
    gaus_flyby_pos_max = np.max(gaus_flyby_pos)
    gaus_flyby_pos_occurences = gaus_flyby_pos/gaus_flyby_pos_max*80
    ax.fill_between(xdata, gaus_flyby_pos_occurences, 0, color='orange', alpha=0.5, edgecolor='none')
    ax2.fill_between(xdata, gaus_flyby_pos_occurences, 0, color='orange', alpha=0.5, edgecolor='none')
    
    # flyby neg gaussian:
    gaus_flyby_neg = np.exp(-1*(xdata-flyby_mean*-1)**2 / (2*flyby_std**2)) * (1/(flyby_std*np.sqrt(2*np.pi)))
    #gaus_flyby_pos_integral = np.sum(gaus_flyby_pos*floris_math.diffa(xdata))
    #print gaus_flyby_pos_integral
    gaus_flyby_neg_max = np.max(gaus_flyby_neg)
    gaus_flyby_neg_occurences = gaus_flyby_neg/gaus_flyby_neg_max*80
    ax.fill_between(xdata, gaus_flyby_neg_occurences, 0, color='orange', alpha=0.5, edgecolor='none')
    ax2.fill_between(xdata, gaus_flyby_neg_occurences, 0, color='orange', alpha=0.5, edgecolor='none')
        
    # landing gaussian:
    gaus_landing = np.exp(-1*(xdata-landing_mean)**2 / (2*landing_std**2)) * (1/(landing_std*np.sqrt(2*np.pi)))
    #gaus_flyby_pos_integral = np.sum(gaus_flyby_pos*floris_math.diffa(xdata))
    #print gaus_flyby_pos_integral
    gaus_landing_max = np.max(gaus_landing)
    gaus_landing_occurences = gaus_landing/gaus_landing_max*24
    ax.fill_between(xdata, gaus_landing_occurences, 0, color='blue', alpha=0.5, edgecolor='none')
    ax2.fill_between(xdata, gaus_landing_occurences, 0, color='blue', alpha=0.5, edgecolor='none')
    
    # sum
    gaus_sum = gaus_flyby_pos_occurences + gaus_flyby_neg_occurences + gaus_landing_occurences
    ax.plot(xdata, gaus_sum, color='red')
        
    # prep plot    
    def prep_plot(ax, after=False, before=False):
        xticks = [-180, -90, 0, 90, 180]
        yticks = [0, 20, 40, 60, 80, 100]

        if after:
            yticks = [0, .0075, .015]
        if before:
            yticks = [0, .005, .01]
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks, smart_bounds=True)
        ax.set_aspect('auto')
        
        ax.set_xlim([np.min(xticks), np.max(xticks)])
        ax.set_ylim([np.min(yticks), np.max(yticks)])
        
        ax.set_xlabel('Angle to post after turn')
        ax.set_ylabel('Number of saccades')
    
    prep_plot(ax)
    prep_plot(ax2)
    prep_plot(axorig)
    
    prep_plot(axbefore, before=True)
    prep_plot(axafter, after=True)
    
    fig.savefig('angle_to_post_after_saccade_histogram.pdf', format='pdf')
    fig2.savefig('angle_to_post_after_saccade_histogram_landing_vs_flyby.pdf', format='pdf')
    figorig.savefig('angle_to_post_after_saccade_histogram_no_symmetry.pdf', format='pdf')
    
    figbefore.savefig('angle_to_post_before_saccade_histogram_post_vs_nopost.pdf', format='pdf')
    figafter.savefig('angle_to_post_after_saccade_histogram_post_vs_nopost.pdf', format='pdf')

def landing_saccade_plot(dataset, edge='none', plot=True):
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    if edge == 'none':
        angle_to_post = dataset.saccades.angle_to_post
    elif edge == 'far':
        angle_to_post = dataset.saccades.angle_to_far_edge
    elif edge == 'near':
        angle_to_post = dataset.saccades.angle_to_near_edge
    turn_angle = dataset.saccades.true_turn_angle
    
    if plot:
        d = 0.1 - dataset.saccades.distance_to_post
        colornorm = matplotlib.colors.Normalize(0,0.1)
        ax.scatter(angle_to_post*180/np.pi, turn_angle*180./np.pi, c=d, linewidth=0, s=3, norm=colornorm)

    # calculate and plot linear regression    
    fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(angle_to_post*180/np.pi, turn_angle*180/np.pi, full_output=True, remove_outliers=True)
    
    xdata = np.arange(-180, 180)
    ydata = np.polyval(fitfmin, xdata)
    ydata_p = std_error*2 + ydata
    ydata_m = -1*std_error*2 + ydata
    
    if plot:
        ax.plot(xdata, ydata, '-', color='black', zorder=-1)
        ax.fill_between(xdata, ydata_p, ydata_m, color='black', alpha=0.3, linewidth=0, zorder=-1)
    
    if plot is False:
        return xdata, ydata, std_error

    # print text of vals
    eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
    rsq = 'Rsq: ' + str(Rsq)[0:5]
    s = eqn + '\n' + rsq
    ax.text( 50, -100, s)

    # prep plot    
    xticks = [-270, -180, -90, 0, 90, 180, 270]
    yticks = [-270, -180, -90, 0, 90, 180, 270]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks, smart_bounds=True)
    ax.set_aspect('equal')
    
    ax.set_xlim([np.min(xticks), np.max(xticks)])
    ax.set_ylim([np.min(yticks), np.max(yticks)])
    
    xlabel = 'Angle to post ' + edge
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Turn angle')
    
    fname = 'landing_saccade_plot_' + str(edge) + '_edge.pdf'
    fig.savefig(fname, format='pdf')
    
    
def flyby_saccade_plot(dataset, dataset_landing=None, edge='none', show_regression=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if edge == 'none':
        angle_to_post = dataset.saccades.angle_to_post
    elif edge == 'far':
        angle_to_post = dataset.saccades.angle_to_far_edge
    elif edge == 'near':
        angle_to_post = dataset.saccades.angle_to_near_edge
    turn_angle = dataset.saccades.true_turn_angle
    
    d = 0.1 - dataset.saccades.distance_to_post
    #d = dataset.saccades.speed
    colornorm = matplotlib.colors.Normalize(0,0.1)
    ax.scatter(angle_to_post*180/np.pi, turn_angle*180/np.pi, c=d, s=3, norm=colornorm, linewidth=0)

    # show landing shading
    if dataset_landing is not None:
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.true_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
        xdata = np.linspace(-180, 180, 10, endpoint=True)
        ydata = np.polyval(fitfmin, xdata)
        ax.fill_between(xdata, ydata+2*std_error, ydata-2*std_error, color='black', alpha=0.3, linewidth=0, zorder=-1)
    
    # show regression
    if show_regression:
        clusters = ['top', 'bottom']
        for cluster in clusters:
            cluster_indices = get_cluster_indices(dataset, dataset_landing, cluster=cluster)
            fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(angle_to_post[cluster_indices]*180/np.pi, turn_angle[cluster_indices]*180/np.pi, full_output=True, remove_outliers=True)
            
            xdata = np.arange(-180, 180)
            ydata = np.polyval(fitfmin, xdata)
            ax.plot(xdata, ydata, '-', color='black')

            # print text of vals
            eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
            rsq = 'Rsq: ' + str(Rsq)[0:5]
            s = eqn + '\n' + rsq
            if cluster == 'top':
                ax.text( -50, 100, s)
            if cluster == 'bottom':
                ax.text( 50, -100, s)

    # prep plot    
    ticks = [-270, -180, -90, 0, 90, 180, 270]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks, smart_bounds=True)
    ax.set_aspect('equal')
    
    ax.deceset_xlim([np.min(ticks), np.max(ticks)])
    ax.set_ylim([np.min(ticks), np.max(ticks)])
    
    xlabel = 'Angle to post'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Turn angle')
    
    fname = 'flyby_saccade_plot_' + str(edge) + '_edge.pdf'
    fig.savefig(fname, format='pdf')

def flyby_saccade_plot_flipsy(dataset, dataset_landing):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    angle_to_post = dataset.saccades.angle_to_post
    turn_angle = dataset.saccades.true_turn_angle
    
    d = 0.1 - dataset.saccades.distance_to_post
    colornorm = matplotlib.colors.Normalize(0,0.1)
    ax.scatter(angle_to_post*180/np.pi, turn_angle*180/np.pi, c=d, s=3, norm=colornorm, linewidth=0.25)
    
    ax.scatter(angle_to_post*180/np.pi-180, turn_angle*180/np.pi, c=d, s=3, norm=colornorm, alpha=0.5, linewidth=0)
    ax.scatter(angle_to_post*180/np.pi+180, turn_angle*180/np.pi, c=d, s=3, norm=colornorm, alpha=0.5, linewidth=0)
    #fpl.scatter(ax, angle_to_post*180/np.pi, turn_angle*180/np.pi+180, color=dataset.saccades.speed, radius=3)

    # flipsy
    indices = np.where(turn_angle < 0)[0].tolist()
    
    #fpl.scatter(ax, angle_to_post[indices]*180/np.pi, turn_angle[indices]*180/np.pi+360, color=dataset.saccades.true_turn_angle[indices], radius=3)
    
    # prep plot    
    ticks = [-270, -180, -90, 0, 90, 180, 270]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks, smart_bounds=True)
    ax.set_aspect('equal')
    
    ax.set_xlim([np.min(ticks), np.max(ticks)])
    ax.set_ylim([np.min(ticks), np.max(ticks)])
    
    xlabel = 'Angle to post flipsy'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Turn angle')
    
    fname = 'flyby_saccade_plot_flipsy.pdf'
    fig.savefig(fname, format='pdf')
    
def get_flyby_saccade_clusters(dataset_flyby, dataset_landing=None):
    
    # landing line
    if dataset_landing is not None:
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.true_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
        xdata_landing = np.linspace(-180, 180, 10, endpoint=True)
        ydata_landing = np.polyval(fitfmin, xdata_landing)
        
    saccades_flipsy = copy.deepcopy(dataset_flyby.saccades)
    indices_adjusted = []
    
    for i in range(len(saccades_flipsy.true_turn_angle)):
        cluster = get_cluster(dataset_flyby.saccades.angle_to_post[i]*180/np.pi, dataset_flyby.saccades.true_turn_angle[i]*180/np.pi, xdata_landing, ydata_landing)
        if cluster == 'top':
            if dataset_flyby.saccades.true_turn_angle[i] < 0:
                saccades_flipsy.angle_to_post[i] += np.pi
                indices_adjusted.append(i)
        if cluster == 'bottom':
            if dataset_flyby.saccades.true_turn_angle[i] > 0:
                saccades_flipsy.angle_to_post[i] -= np.pi
                indices_adjusted.append(i)
    indices_adjusted = set(indices_adjusted)
                
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
                
    # now get new clusters:
    clusters = ['top', 'bottom']
    color = ['red', 'blue']
    for i, cluster in enumerate(clusters):
        indices = get_cluster_indices(saccades_flipsy, dataset_landing, cluster=cluster)
        ax.scatter(saccades_flipsy.angle_to_post[indices]*180/np.pi, saccades_flipsy.true_turn_angle[indices]*180/np.pi, c=color[i], s=3, linewidth=0)
        
        # show the adjustments that were made to the data
        adjusted_indices_to_show = sorted( indices_adjusted.intersection(indices) )
        
        
        ax.scatter(dataset_flyby.saccades.angle_to_post[adjusted_indices_to_show]*180/np.pi, dataset_flyby.saccades.true_turn_angle[adjusted_indices_to_show]*180/np.pi, c=color[i], s=3, linewidth=0, alpha=0.5)
        ax.scatter(saccades_flipsy.angle_to_post[adjusted_indices_to_show]*180/np.pi, saccades_flipsy.true_turn_angle[adjusted_indices_to_show]*180/np.pi, c=color[i], s=3, linewidth=0.5)
        
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type2(saccades_flipsy.angle_to_post[indices]*180/np.pi, saccades_flipsy.true_turn_angle[indices]*180/np.pi, full_output=True, remove_outliers=True)
        
        xdata = np.arange(-180, 180)
        ydata = np.polyval(fitfmin, xdata)
        ax.plot(xdata, ydata, '-', color=color[i])

        # print text of vals
        eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
        rsq = 'Rsq: ' + str(Rsq)[0:5]
        s = eqn + '\n' + rsq
        if cluster == 'top':
            ax.text( -50, 100, s, color=color[i])
        if cluster == 'bottom':
            ax.text( 50, -100, s, color=color[i])
    # prep plot    
    ticks = [-270, -180, -90, 0, 90, 180, 270]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks, smart_bounds=True)
    ax.set_aspect('equal')
    
    ax.set_xlim([np.min(ticks), np.max(ticks)])
    ax.set_ylim([np.min(ticks), np.max(ticks)])
    
    xlabel = 'Angle to post'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Turn angle')
    fig.savefig('flyby_saccades_flipsy_clusters.pdf', format='pdf')
    

def histogram_of_toward_away_from_post(dataset_flyby, dataset_landing):
    # landing line
    if dataset_landing is not None:
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.true_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
        xdata_landing = np.linspace(-180, 180, 10, endpoint=True)
        ydata_landing = np.polyval(fitfmin, xdata_landing)
    
    saccades = dataset_flyby.saccades
    
    # now get new clusters:
    left_indices = np.where(saccades.true_turn_angle > 0)[0].tolist()
    right_indices = np.where(saccades.true_turn_angle < 0)[0].tolist()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    turn_angle = saccades.true_turn_angle*180/np.pi
    angle_to_post = saccades.angle_to_post*180/np.pi
    
    bins = np.linspace(-180, 180, 25, endpoint=True)
    bins, data_hist_list, data_curve_list = fpl.histogram(ax, [angle_to_post[left_indices], angle_to_post[right_indices]], colors=['red', 'blue'], bins=20, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=True, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    
    left_hist = np.abs(data_curve_list[0])
    right_hist = np.abs(data_curve_list[1])
    eps = 0.00001
    ratio = left_hist / (right_hist + left_hist + eps)
    print '*', right_hist[0], left_hist[0]
        
    # prep plot    
    xticks = [-270, -180, -90, 0, 90, 180, 270]
    yticks = [0, 0.005, 0.01, 0.015]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks, smart_bounds=True)
    
    bin_centers = np.diff(bins)/2. + bins[0:-1]
    indices = np.where( np.abs(bin_centers) < 91)[0].tolist()
    # plot ratio
    #ax.plot(bin_centers[indices], ratio[indices]*.02, 'red')
    
    ax.set_xlim([np.min(xticks), np.max(xticks)])
    ax.set_ylim([np.min(yticks), np.max(yticks)])
    
    xlabel = 'Angle to post'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Occurences')
    fig.savefig('flyby_saccades_towards_or_away.pdf', format='pdf')
    

def histogram_of_amplitude(dataset_flyby, dataset_landing):    
    # landing line
    if dataset_landing is not None:
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.true_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
        xdata_landing = np.linspace(-180, 180, 10, endpoint=True)
        ydata_landing = np.polyval(fitfmin, xdata_landing)
    
    saccades = dataset_flyby.saccades
    
    # now get new clusters:
    left_indices = np.where(saccades.true_turn_angle > 0)[0].tolist()
    right_indices = np.where(saccades.true_turn_angle < 0)[0].tolist()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    turn_angle = saccades.true_turn_angle*180/np.pi
    
    bins = np.linspace(-270, 270, 30, endpoint=True)
    bins, data_hist_list, data_curve_list = fpl.histogram(ax, [turn_angle[left_indices], turn_angle[right_indices]], colors=['red', 'blue'], bins=bins, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=True, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    
        
    # prep plot    
    xticks = [-270, -180, -90, 0, 90, 180, 270]
    yticks = [0, 0.005, 0.01, 0.015]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks, smart_bounds=True)
    
    ax.set_xlim([np.min(xticks), np.max(xticks)])
    ax.set_ylim([np.min(yticks), np.max(yticks)])
    
    xlabel = 'Turn Angle'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Occurences')
    fig.savefig('flyby_saccades_amplitude.pdf', format='pdf')
    
    

def get_flyby_saccade_clusters_for_merged_clusters(dataset_flyby, dataset_landing=None):
    
    # landing line
    if dataset_landing is not None:
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(dataset_landing.saccades.angle_to_post*180/np.pi, dataset_landing.saccades.true_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
        xdata_landing = np.linspace(-180, 180, 10, endpoint=True)
        ydata_landing = np.polyval(fitfmin, xdata_landing)
        
    saccades_flipsy = copy.deepcopy(dataset_flyby.saccades)
    indices_adjusted = []
    
    for i in range(len(saccades_flipsy.true_turn_angle)):
        cluster = get_cluster(dataset_flyby.saccades.angle_to_post[i]*180/np.pi, dataset_flyby.saccades.true_turn_angle[i]*180/np.pi, xdata_landing, ydata_landing)
        if cluster == 'top':
            if dataset_flyby.saccades.true_turn_angle[i] < 0:
                saccades_flipsy.angle_to_post[i] += np.pi
                indices_adjusted.append(i)
        if cluster == 'bottom':
            if dataset_flyby.saccades.true_turn_angle[i] > 0:
                saccades_flipsy.angle_to_post[i] -= np.pi
                indices_adjusted.append(i)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(saccades_flipsy.angle_to_post*180/np.pi, saccades_flipsy.true_turn_angle*180/np.pi, c='black', s=3, linewidth=0, alpha=0.3)
    
    for i in range(len(saccades_flipsy.true_turn_angle)):
        cluster = get_cluster(saccades_flipsy.angle_to_post[i]*180/np.pi, saccades_flipsy.true_turn_angle[i]*180/np.pi, xdata_landing, ydata_landing)
        if cluster == 'top':
            saccades_flipsy.angle_to_post[i] += np.pi
    
    
    ax.scatter(saccades_flipsy.angle_to_post*180/np.pi, saccades_flipsy.true_turn_angle*180/np.pi, c='black', s=3, linewidth=0)
        
    fitfmin, std_error, Rsq = linear_fit.linear_fit_type2(saccades_flipsy.angle_to_post*180/np.pi, saccades_flipsy.true_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
    
    xdata = np.arange(-180, 270)
    ydata = np.polyval(fitfmin, xdata)
    ax.plot(xdata, ydata, '-', color='black')

    # print text of vals
    eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
    rsq = 'Rsq: ' + str(Rsq)[0:5]
    s = eqn + '\n' + rsq
    ax.text( -50, 100, s, color='black')
    # prep plot    
    ticks = [-270, -180, -90, 0, 90, 180, 270]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks, smart_bounds=True)
    ax.set_aspect('equal')
    
    ax.set_xlim([np.min(ticks), np.max(ticks)])
    ax.set_ylim([np.min(ticks), np.max(ticks)])
    
    xlabel = 'Angle to post'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Turn angle')
    fig.savefig('flyby_saccades_flipsy_clusters_merged.pdf', format='pdf')
    



def get_cluster(x, y, xdata_landing, ydata_landing):
    y_check = np.interp(x, xdata_landing, ydata_landing)
    if y > y_check:
        return 'top'
    elif y < y_check:
        return 'bottom'
    
    
def is_point_evasive(x, y, xdata, ydata_p, ydata_m):  
    y_check_p = np.interp(x, xdata, ydata_p)
    y_check_m = np.interp(x, xdata, ydata_m)
    if y > y_check_p or y < y_check_m:
        return True
    else:
        return False
        
def get_evasive_and_attractive_indices(dataset_flyby, dataset_landing):
    evasive_indices = []
    attractive_indices = []
    xdata, ydata, std_error = landing_saccade_plot(dataset_landing, plot=False)
    ydata_p = ydata + 2*std_error 
    ydata_m = ydata - 2*std_error
    for i, s in enumerate(dataset_flyby.saccades.angle_to_post):
        x = dataset_flyby.saccades.angle_to_post[i]*180./np.pi
        y = dataset_flyby.saccades.true_turn_angle[i]*180./np.pi
        is_evasive = is_point_evasive(x,y, xdata, ydata_p, ydata_m)
        if is_evasive:
            evasive_indices.append(i)
        else:
            attractive_indices.append(i)
            
    return evasive_indices, attractive_indices
    

        
def get_cluster_indices(saccades, dataset_landing, cluster='top'):
    indices = []
    xdata_landing, ydata_landing, std_error = landing_saccade_plot(dataset_landing, plot=False)
    ydata_p = ydata_landing + 2*std_error 
    ydata_m = ydata_landing - 2*std_error
    for i, turn_angle in enumerate(saccades.true_turn_angle):
        x = saccades.angle_to_post[i]*180./np.pi
        y = saccades.true_turn_angle[i]*180./np.pi
        which_cluster = get_cluster(x, y, xdata_landing, ydata_landing)
        if which_cluster == cluster:
            indices.append(i)
    return indices

            
def flyby_histograms(dataset_flyby, dataset_landing):
    evasive_indices, attractive_indices = get_evasive_and_attractive_indices(dataset_flyby, dataset_landing)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, [np.log(dataset_flyby.saccades.angle_subtended_by_post)], colors=['orange'], bins=20, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.5)
    
    yticks = np.linspace(0, 200, 3, endpoint=True)
    set_log_ticks(ax, yticks, radians=True)
    
    fig.savefig('flyby_histograms.pdf', format='pdf')
    
def landing_histograms(dataset_landing):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, [np.log(dataset_landing.saccades.angle_subtended_by_post)], colors=['blue'], bins=20, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.5)
    
    yticks = np.linspace(0, 24, 3, endpoint=True)
    set_log_ticks(ax, yticks, radians=True)
    
    fig.savefig('landing_histograms.pdf', format='pdf')
    
    
def saccade_distance_histograms(dataset_flyby, dataset_landing, dataset_nopost_flyby, dataset_nopost_landing):

    withpost = np.hstack([dataset_flyby.saccades.angle_subtended_by_post, dataset_landing.saccades.angle_subtended_by_post])
    nopost = np.hstack([dataset_nopost_flyby.saccades.angle_subtended_by_post, dataset_nopost_landing.saccades.angle_subtended_by_post])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram(ax, [np.log(withpost), np.log(nopost)], colors=['black', 'green'], bins=24, edgecolor='none', bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
    
    yticks = np.linspace(0, 0.8, 3, endpoint=True)
    set_log_ticks(ax, yticks, radians=True)
    
    fig.savefig('saccade_distance_histograms.pdf', format='pdf')
    
    
    


def set_log_ticks(ax, yticks=None, radians=False):
    
    deg_ticks = np.array([5, 10, 30, 60, 90, 180])
    deg_tick_strings = [str(d) for d in deg_ticks]
    if radians:
        xticks = deg_ticks*np.pi/180.
    else:
        xticks = deg_ticks
    xticks_log = np.log(xticks)
    dist_tick_strings = ['(21)', '(10)', '(2.7)', '(0.9)', '(0.4)', '(0)']
    
    x_tick_strings = []
    for i, d in enumerate(dist_tick_strings):
        x_tick_strings.append( deg_tick_strings[i] + '\n' + dist_tick_strings[i] )
    
    ax.set_xlim(xticks_log[0], xticks_log[-1])
    if yticks is not None:
        ax.set_ylim(yticks[0], yticks[-1])
        
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks_log, yticks=yticks)        
        
    ax.set_xticklabels(x_tick_strings)
    ax.set_xlabel('Retinal size\n(Distance to post, cm)')
    ax.set_ylabel('Occurences')
    
    
def left_right_turn_probability(dataset_flyby):
    
    #indices_left = np.where(dataset_flyby.saccades
    #indices_right = []
    pass
    
    
def speed_vs_turn_angle(dataset, colorscale=True):
    fig = plt.figure()
    ax= fig.add_subplot(111)
    diff_in_turn_angle = floris_math.fix_angular_rollover(dataset.saccades.true_turn_angle) / floris_math.fix_angular_rollover(dataset.saccades.true_turn_angle)
    #diff_in_turn_angle = np.abs(floris_math.fix_angular_rollover(diff_in_turn_angle))
    
    indices = np.where( np.abs(dataset.saccades.angle_to_post) < 60*np.pi/180.)[0].tolist()
    
    #x = (np.abs(dataset.saccades.expansion) / dataset.saccades.angle_subtended_by_post)[indices]
    x = diff_in_turn_angle[indices]
    y = np.abs(dataset.saccades.expansion[indices]) / dataset.saccades.angle_subtended_by_post[indices]
    
    if colorscale:
        ax.scatter( x, y, c=dataset.saccades.angle_to_post[indices], cmap='jet', s=2, linewidth=0)
    else:
        ax.scatter( x, y, c='black', s=2, linewidth=0)
        
    # linear fit
    # calculate and plot linear regression    
    fitfmin, std_error, Rsq = linear_fit.linear_fit_type2(x, y, full_output=True, remove_outliers=False)
    
    xdata = np.linspace(0, 40, 10, endpoint=True)
    ydata = np.polyval(fitfmin, xdata)
    ax.plot(xdata, ydata, '-', color='black')
    
    # print text of vals
    eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
    rsq = 'Rsq: ' + str(Rsq)[0:5]
    s = eqn + '\n' + rsq
    ax.text( 10, 0.5, s, color='purple')
    

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 25)
    fpl.adjust_spines(ax, ['left', 'bottom'])
    fig.savefig('speed_vs_turn_angle.pdf', format='pdf')
    

def landing_deceleration_plot(dataset, show_regression=True, show_traces=False, keys=None, show_histograms=False, show_where_saccade_after_decel=False):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    speeds = []
    angles_subtended = []
    filtered_keys = []
    angles_to_post = []
    keys_where_saccade_after_decel = []
    
    saccade_angles_to_post = []
    saccade_turn_angles = []

    def plot(trajec):
        fd = trajec.frame_at_deceleration
        #ax.plot( np.log(trajec.angle_subtended_by_post[fd]), trajec.speed[fd], '.', color='purple', linewidth=0, markersize=3)
        speeds.append(trajec.speed[fd])
        angles_subtended.append(np.log(trajec.angle_subtended_by_post[fd]))
        angles_to_post.append(trajec.angle_to_post[fd]*180./np.pi)
        filtered_keys.append(trajec.key)

        if len(trajec.last_saccade_range) > 0:
            tmp = floris_math.fix_angular_rollover(trajec.angle_to_post[trajec.last_saccade_range[0]]-trajec.angle_to_post[trajec.last_saccade_range[-1]])
            saccade_turn_angles.append(tmp)
            saccade_angles_to_post.append(trajec.angle_to_post[trajec.last_saccade_range[0]])
        
    for k, trajec in dataset.trajecs.items():
        if trajec.frame_at_deceleration is not None:
            fd = trajec.frame_at_deceleration
            if len(trajec.last_saccade_range) > 0:
                if trajec.frame_at_deceleration > trajec.last_saccade_range[-1]:
                    plot(trajec)
                else:
                    keys_where_saccade_after_decel.append(k)
            else:
                plot(trajec)
                
                
    speeds = np.array(speeds)
    angles_subtended = np.array(angles_subtended) # this has already been through the log operator!!
    angles_to_post = np.array(angles_to_post)

    ax.scatter(angles_subtended, speeds, color='purple', s=7, linewidth=0)
    
    if show_regression:
        # calculate and plot linear regression    
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(angles_subtended, speeds, full_output=True, remove_outliers=True)
        
        xdata = np.linspace(np.log(5*np.pi/180.), np.log(180*np.pi/180.), 10, endpoint=True)
        ydata = np.polyval(fitfmin, xdata)
        ax.plot(xdata, ydata, '-', color='purple')
        
        # print text of vals
        eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
        rsq = 'Rsq: ' + str(Rsq)[0:5]
        s = eqn + '\n' + rsq + '\n' + 'N: ' + str(len(angles_subtended))
        ax.text( np.log(10*np.pi/180.), 0.3, s, color='purple')
        
    if 1: #show_where_saccade_after_decel:
        time_between_decel_and_saccade = []
        angles_sd = []
        speeds_sd = []
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        for key in keys_where_saccade_after_decel:
            trajec = dataset.trajecs[key]
            fd = trajec.frame_at_deceleration
            fs = trajec.last_saccade_range[0]
            #ax.plot(np.log(trajec.angle_subtended_by_post[fs]), trajec.speed[fs], '.', color='red', linewidth=0, markersize=3)
            #ax.plot([np.log(trajec.angle_subtended_by_post[fd]), np.log(trajec.angle_subtended_by_post[fs])], [trajec.speed[fd],trajec.speed[fs]], '-', color='black')
            time_between_decel_and_saccade.append( float(fs-fd)/trajec.fps )
            
            if len(trajec.last_saccade_range) > 0:
                tmp = floris_math.fix_angular_rollover(trajec.angle_to_post[trajec.last_saccade_range[0]]-trajec.angle_to_post[trajec.last_saccade_range[-1]])
                ax2.plot(trajec.angle_to_post[trajec.last_saccade_range[0]]*180./np.pi, tmp*180./np.pi, '.', linewidth=0, color='blue', markersize=3)
                
            angles_sd.append(np.log(trajec.angle_subtended_by_post[fd]))
            speeds_sd.append(trajec.speed[fd])
        angles_sd = np.array(angles_sd)
        speeds_sd = np.array(speeds_sd)
        
        if show_where_saccade_after_decel:
            ax.scatter(angles_sd, speeds_sd, color='blue', linewidth=0, s=7)
        
        print
        print 'mean time between fs and fd: ', np.mean(np.array(time_between_decel_and_saccade))
        print 'std: ', np.std(np.array(time_between_decel_and_saccade))

        ######### for saccade plot ##########
        # show the other saccades too?
        ax2.plot(dataset.saccades.angle_to_post*180./np.pi, dataset.saccades.true_turn_angle*180./np.pi, '.', linewidth=0, color='purple', markersize=3, zorder=-1)
        
        # prep plot    
        ticks = [-180, -90, 0, 90, 180]
        fpl.adjust_spines(ax2, ['left', 'bottom'], xticks=ticks, yticks=ticks, smart_bounds=True)
        ax2.set_aspect('equal')
        
        ax2.set_xlim([-180, 180])
        ax2.set_ylim([-180, 180])
        
        xlabel = 'Angle to post'
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Turn angle')
        fig2.savefig('landing_saccade_plot_where_saccade_after_deceleration.pdf', format='pdf')
        #####################################
        
        if show_regression:
            # calculate and plot linear regression    
            fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(angles_sd, speeds_sd, full_output=True, remove_outliers=True)
            
            xdata = np.linspace(np.log(5*np.pi/180.), np.log(180*np.pi/180.), 10, endpoint=True)
            ydata = np.polyval(fitfmin, xdata)
            ax.plot(xdata, ydata, '-', color='blue')
            
            # print text of vals
            eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
            rsq = 'Rsq: ' + str(Rsq)[0:5]
            s = eqn + '\n' + rsq + '\n' + 'N: ' + str(len(angles_sd))
            ax.text( np.log(40*np.pi/180.), 0.7, s, color='blue')
            
        
        
            

    if show_histograms:
        fpl.histogram(ax, [angles_subtended, angles_sd], colors=['maroon', 'lightblue'], bins=20, edgecolor='none', bar_alpha=1, curve_fill_alpha=0, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
        #fpl.histogram(ax, [np.log(angles_subtended)], colors=['purple'], bins=20, bar_alpha=1, curve_fill_alpha=0.3, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=False, normed=False, normed_occurences=False, bootstrap_std=False, exponential_histogram=False, bin_width_ratio=0.8)
        
    
    if show_traces:
        print
        print 'keys to highlight: '
        if keys == None:
            keys = ['2_31060', '2_3954', '8_10258', '1_28622', '6_18259']
        if keys == 'all':
            keys = filtered_keys
        for key in keys:
            print key
            trajec = dataset.trajecs[key]
            indices = np.arange(trajec.frame_at_deceleration-25, trajec.frame_of_landing).tolist()
            angles = np.log(trajec.angle_subtended_by_post[indices])
            speeds = trajec.speed[indices]
            ax.plot(angles, speeds, '-', color='black', zorder=-1)
        
    yticks = np.linspace(0,1,5,endpoint=True)
    set_log_ticks(ax, yticks=yticks, radians=True)
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
                
                
                
def flyby_deceleration_plot(dataset, show_traces=False, keys=None, show_histograms=False, show_regression=True):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    speeds = []
    angles_subtended = []
    filtered_keys = []

    def plot(trajec):
        fd = trajec.frame_at_deceleration
        ax.scatter( np.log(trajec.angle_subtended_by_post[fd]), trajec.speed[fd], c='purple', s=7, linewidth=0)
        speeds.append(trajec.speed[fd])
        angles_subtended.append(np.log(trajec.angle_subtended_by_post[fd]))
        filtered_keys.append(trajec.key)
        
    for k, trajec in dataset.trajecs.items():
        if trajec.frame_at_deceleration is not None:
            angle_to_post_at_deceleration = np.abs(trajec.angle_to_post[trajec.frame_at_deceleration])*180./np.pi
            if angle_to_post_at_deceleration < 43:
                fd = trajec.frame_at_deceleration
                if len(trajec.last_saccade_range) > 0:
                    if trajec.frame_at_deceleration > trajec.last_saccade_range[-1]:
                        plot(trajec)
                    else:
                        pass
                else:
                    plot(trajec)
                
    speeds = np.array(speeds)
    angles_subtended = np.array(angles_subtended) # this has already been through the log operator!!
                
    if show_regression:
        # calculate and plot linear regression    
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(angles_subtended, speeds, full_output=True, remove_outliers=True)
        
        xdata = np.linspace(np.log(5*np.pi/180.), np.log(180*np.pi/180.), 10, endpoint=True)
        ydata = np.polyval(fitfmin, xdata)
        ax.plot(xdata, ydata, '-', color='purple')
        
        # print text of vals
        eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
        rsq = 'Rsq: ' + str(Rsq)[0:5]
        s = eqn + '\n' + rsq + '\n' + 'N: ' + str(len(angles_subtended))
        ax.text( np.log(10*np.pi/180.), 0.3, s)
        
    yticks = np.linspace(0,1,5,endpoint=True)
    set_log_ticks(ax, yticks=yticks, radians=True)
    ax.set_ylabel('Speed, m/s')
                
    fig.savefig('flyby_deceleration_plot.pdf', format='pdf')



def heatmap_landing_trajecs(dataset_landing, dataset_flyby):

    speeds = []
    angles_subtended = []
    for k, trajec in dataset_landing.trajecs.items():
        speeds.extend(trajec.speed[0:trajec.frame_of_landing].tolist())
        angles_subtended.extend(trajec.angle_subtended_by_post[0:trajec.frame_of_landing].tolist())
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fpl.histogram2d(ax, np.log(np.array(angles_subtended)), np.array(speeds), bins=50, normed=False, histrange=None, weights=None, logcolorscale=True, colormap='jet', interpolation='bicubic')
    
    # now look at speed and retinal size for flyby trajectories at last saccade if headed towards post:
    for k, trajec in dataset_flyby.trajecs.items():
        if len(trajec.last_saccade_range) > 0: 
            fs = trajec.last_saccade_range[0]
            ax.plot(np.log(trajec.angle_subtended_by_post[fs]), trajec.speed[fs], '.', markersize=3, color='white', markeredgecolor='black', linewidth=0.5)
    
    ax.set_xlim(np.log(5*np.pi/180.), np.log(180*np.pi/180.))
    ax.set_ylim(0,1)
    ax.set_aspect('auto')
    yticks = np.linspace(0,1,5,endpoint=True)
    set_log_ticks(ax, yticks=yticks, radians=True)        
    ax.set_ylabel('Speed, m/s')
    
    fig.savefig('landing_deceleration_heatmap.pdf')
    
    
def get_true_turn_angle_fit_from_speed_and_rel_turn_angle(dataset):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    tmp = dataset.saccades.speed
    ax.scatter( (dataset.saccades.true_turn_angle)*180./np.pi, (dataset.saccades.true_turn_angle)*180./np.pi, c=tmp, linewidth=0, s=3)
    
    ticks = [-270, -180, -90, 0, 90, 180, 270]
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks)
    
    ax.set_xlim([np.min(ticks), np.max(ticks)])
    ax.set_ylim([np.min(ticks), np.max(ticks)])
    
    ax.set_xlabel('Relative turn angle, deg')
    ax.set_ylabel('True turn angle, deg')
    
    fig.savefig('turn_angle_comparison.pdf', format='pdf')
    
    
    # mirrored:
    x = np.abs( floris_math.fix_angular_rollover(dataset.saccades.true_turn_angle) )
    z = np.abs( floris_math.fix_angular_rollover(dataset.saccades.true_turn_angle) )
    y = dataset.saccades.speed
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter( x, y, c=z, linewidth=0, s=3, zorder=10)
    fig.savefig('turn_angle_comparison_mirrored.pdf', format='pdf')
    
    
    ############
    # Fit a 3rd order, 2d polynomial
    m = fit2dpolynomial.polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = fit2dpolynomial.polyval2d(xx, yy, m)
    
    print m

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(zz, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
    ax.scatter(x, y, c=z, linewidth=0.1, s=2)
    
    ax.set_aspect('auto')
    fpl.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xlabel('Relative turn angle, deg')
    ax.set_ylabel('Speed, m/s')
    fig.savefig('turn_angle_comparison_fit.pdf', format='pdf')
    
    return m    
    
    
    
def plot_true_turn_angle_as_function_of_postangle_and_speed(dataset, dataset_landing):
    
    angle_to_post = dataset.saccades.angle_to_post
    rel_turn_angle = np.abs( floris_math.fix_angular_rollover(dataset.saccades.true_turn_angle) )
    z = np.abs( floris_math.fix_angular_rollover(dataset.saccades.true_turn_angle) )
    speed = dataset.saccades.speed
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    clusters = ['top', 'bottom']
    for cluster in clusters:
        cluster_indices = get_cluster_indices(dataset, dataset_landing, cluster=cluster)
        fit_rel_turn_angle_from_post_angle, std_error, Rsq = linear_fit.linear_fit_type1(dataset.saccades.angle_to_post[cluster_indices], dataset.saccades.true_turn_angle[cluster_indices], full_output=True, remove_outliers=True)
        
        figtmp = plt.figure()
        axtmp = figtmp.add_subplot(111)
        axtmp.plot(dataset.saccades.angle_to_post[cluster_indices], dataset.saccades.true_turn_angle[cluster_indices], '.')
        figtmp.savefig('tmp.pdf', format='pdf')
        
        print 'Rsq: ', Rsq
        
        m = get_true_turn_angle_fit_from_speed_and_rel_turn_angle(dataset)
    
        est_rel_turn_angle = np.polyval(fit_rel_turn_angle_from_post_angle, angle_to_post[cluster_indices])
        
        
        x = est_rel_turn_angle
        y = speed[cluster_indices]
    
        # Evaluate it on a grid...
        if 0:
            nx, ny = 20, 20
            xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                                 np.linspace(y.min(), y.max(), ny))
            est_turn_angle = fit2dpolynomial.polyval2d(xx, yy, m)
        
        est_turn_angle = fit2dpolynomial.polyval2d( np.abs(floris_math.fix_angular_rollover(x)), y, m)
        
        # Plot
        norm = matplotlib.colors.Normalize(0,np.pi)
        
            
        ax.plot(z[cluster_indices]*180/np.pi, est_turn_angle*180/np.pi, '.')

        # calculate and plot linear regression    
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type1(z[cluster_indices]*180/np.pi, est_turn_angle*180/np.pi, full_output=True, remove_outliers=True)
        
        xdata = np.arange(0, 180)
        ydata = np.polyval(fitfmin, xdata)
        ydata_p = std_error*2 + ydata
        ydata_m = -1*std_error*2 + ydata
        
        ax.plot(xdata, ydata, '-', color='black')
        ax.fill_between(xdata, ydata_p, ydata_m, color='black', alpha=0.3, linewidth=0)
        
        # print text of vals
        eqn = 'y = ' + str(fitfmin[0])[0:5] + 'x + ' + str(fitfmin[1])[0:5]
        rsq = 'Rsq: ' + str(Rsq)[0:5]
        s = eqn + '\n' + rsq
        ax.text( 50, -100, s)
        
        ax.set_aspect('equal') 
        ticks = [0, 90, 180]
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=ticks, yticks=ticks)
        
        ax.set_xlim(np.min(ticks), np.max(ticks))
        ax.set_ylim(np.min(ticks), np.max(ticks))
        
    ax.set_xlabel('Actual turn angle, deg')
    ax.set_ylabel('Predicted turn angle, deg')
    fig.savefig('est_vs_true_turn_angle.pdf', format='pdf')        
        
        
        
    
    data, bins = np.histogram( np.log(angle_at_leg_extension), bins=16, normed=True)
    xvals = np.diff(bins) + bins[0:-1]
    
    #butter_b, butter_a = signal.butter(3, 0.3)
    #data_filtered = signal.filtfilt(butter_b, butter_a, data)
    
    print 'N = ', n
    print 'mean angle: ', np.mean(angle_at_leg_extension)*180/np.pi
    
    if plot and post_type_color is False:
        fig = plt.figure()
        fig.set_facecolor('white')
        ax = fig.add_subplot(111) 
        #ax.hist( np.log(angle_at_leg_extension), bins=bins, normed=True, facecolor='green', alpha=0.1, edgecolor='green')
        bins, hists, hist_std, curves = fpl.histogram(ax, [np.log(angle_at_leg_extension)], bins=bins, colors='green', bin_width_ratio=0.8, edgecolor='none', bar_alpha=0.2, curve_fill_alpha=0, curve_line_alpha=0.8, return_vals=True, normed_occurences='total', bootstrap_std=False)
        
        
        #ax.plot( xvals, data_filtered, color='green' )     
        ax.scatter( np.log(angle_at_leg_extension), speed_at_leg_extension, s=3, c='green', linewidth=0)
        
        #fit = np.polyfit( np.log(angle_at_leg_extension), speed_at_leg_extension, 1)
        print np.log(angle_at_leg_extension).shape, speed_at_leg_extension.shape
        fitfmin, std_error, Rsq = linear_fit.linear_fit_type2(np.log(angle_at_leg_extension), speed_at_leg_extension, full_output=True)
        x = np.linspace(np.min(np.log(angle_at_leg_extension)), np.max(np.log(angle_at_leg_extension)), 100)
        y = np.polyval(fit, x)
        ax.plot(x,y,color='green')
        
        yplus = np.polyval(fit, x+np.sqrt(variance))+np.sqrt(variance)
        yminus = np.polyval(fit, x-np.sqrt(variance))-np.sqrt(variance)
        #ax.fill_between(x, yplus, yminus, color='green', linewidth=0, alpha=0.2)
    
        slope = str(fit[0])[0:5]
        intercept = str(fit[1])[0:5]
        string = 'y = ' + slope + 'x + ' + intercept
        ax.text(0, 0.5, string)
        string = 'Rsq=' + str(Rsq)[0:4]
        ax.text(0,0.4, string)
        
        fix_angle_log_spine(ax)
        plt.show()
        string = 'N=' + str(n)
        ax.text( -2, 0.8, string, color='green')
        fig.savefig('leg_ext_histogram.pdf', format='pdf')
        
    if plot and post_type_color is True:
        fig = plt.figure()
        fig.set_facecolor('white')
        ax = fig.add_subplot(111) 
        #ax.hist( np.log(angle_at_leg_extension), bins=bins, normed=True, facecolor='green', alpha=0.1, edgecolor='green')
        bins, hists, hist_std, curves = floris.histogram(ax, [np.log(angle_at_leg_extension[black_post]), np.log(angle_at_leg_extension[checkered_post])], bins=bins, colors=['black', 'teal'], bin_width_ratio=0.9, edgecolor='none', bar_alpha=0.2, curve_fill_alpha=0, curve_line_alpha=0.8, return_vals=True, normed_occurences=True, bootstrap_std=True)
        
        #ax.plot( xvals, data_filtered, color='green' )     
        ax.plot( np.log(angle_at_leg_extension[black_post]), speed_at_leg_extension[black_post], '.', color='black')
        ax.plot( np.log(angle_at_leg_extension[checkered_post]), speed_at_leg_extension[checkered_post], '.', color='teal')
        
        
        fix_angle_log_spine(ax)
        plt.show()
        fig.savefig('leg_ext_histogram_post_types.pdf', format='pdf')
        
    ####  time to touchdown after leg extension
    time_to_landing *= 1000
    print 'n for touchdowns = ', len(time_to_landing)
    print 'n > 0 = ', len(np.where(time_to_landing > 0)[0].tolist())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins, hists, curves = floris.histogram(ax, [np.log(time_to_landing)], bins=15, colors='black', bin_width_ratio=0.9, edgecolor='none', bar_alpha=0.8, curve_fill_alpha=0, curve_line_alpha=0, return_vals=True, show_smoothed=False)
    ax.set_ylim(0,np.max(hists))
    ax.set_xlim(0,6)
    ax.set_autoscale_on(False)
    adjust_spines(ax, ['left', 'bottom'])
    
    xticks = ax.get_xticks()
    xticks_exp = np.exp(xticks)
    
    xtick_labels = [str(x) for x in xticks_exp]
    for i, s in enumerate(xtick_labels):
        tmp = s.split('.')
        xtick_labels[i] = tmp[0]
    
    #ax.set_xticks(xticks_log)
    ax.set_xticklabels(xtick_labels)
    
    ax.set_xlabel('Time to touchdown after leg extension, ms')
    ax.set_ylabel('Occurences')
    filename = 'time_to_touchdown_at_leg_ext.pdf'
    fig.savefig(filename, format='pdf')
    print 'mean time to touchdown after leg ext: ', np.mean(time_to_landing), np.std(time_to_landing)
    print 'n flies that stick out legs less than 50 ms away: ', len(np.where(time_to_landing<100)[0].tolist()) / float( len(time_to_landing) )
    
    return angle_at_leg_extension, bins, data_filtered, xvals
