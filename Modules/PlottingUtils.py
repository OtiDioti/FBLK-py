import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as tkr
from matplotlib.ticker import AutoMinorLocator
import plotly.graph_objects as go
import plotly.io as pio
#%% Colors I like
RdBu = plt.cm.get_cmap('RdBu')
BuRd = RdBu.reversed()
#%%
def HeatMap(A, lowx, highx, lowy, highy, 
             xlabel = '', ylabel = '', zlabel = '', 
             colmap = BuRd, figursize = (8, 8), fntsize = 32, fntscale = 0.7, nbar_ticks = 10, 
             point = False, pointx = 0, pointy = 0, pointlbl = '0', pointclr = 'red', 
             lineclr = 'blue', linewid = 1, linestyl = 'solid', hlinelbl = '0', vlinelbl = '0', 
             dotsperinch = 100, origo = 'lower', aspetto = 'auto'):

    fig = plt.figure(figsize = figursize, dpi = dotsperinch)
    heatmap = plt.imshow(A, origin = origo, extent=[lowx, highx, lowy, highy], cmap=colmap, aspect = aspetto)
    
    # plt.colorbar(label = zlabel)
    cbar = fig.colorbar(heatmap, format=tkr.FormatStrFormatter('%.2g')) 
    cbar.set_label(zlabel, size=fntsize * fntscale)           
    plt.yticks(fontsize=fntsize*fntscale)
    plt.xticks(fontsize=fntsize*fntscale)
    tick_locator = ticker.MaxNLocator(nbins=nbar_ticks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize= fntsize * fntscale)
    
    plt.xlabel(xlabel, fontsize=fntsize)
    plt.ylabel(ylabel, fontsize=fntsize)
    
    if point == True:
        plt.scatter(pointx, pointy, c = pointclr, label = pointlbl )
        plt.vlines(pointx, lowy, highy, linestyles = linestyl, colors = lineclr, label = vlinelbl, linewidth = linewid)
        plt.hlines(pointy, lowx, highx, linestyles = linestyl, colors = lineclr, label = hlinelbl, linewidth = linewid)
        plt.legend()
    else:
        pass

def LinePlot(xdata, ydata, multiple_lines = False, xlbl = '', ylbl = '', xlims = None, ylims = None, 
             title = '', fntsize = 35, fntweight = 'bold', label = None, colors = None, legendloc = 2, 
             legend_size_factor = 1, ticks_size_factor = 0.6, linewidth = 2):
    plt.style.use('classic')
    ax = plt.axes()
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(xlbl, fontsize = fntsize, fontweight = fntweight)
    ax.set_ylabel(ylbl, fontsize = fntsize, fontweight = fntweight)
    
    if xlims != None:   
        ax.set_xlim(xlims[0], xlims[1])
    if ylims != None:    
        ax.set_ylim(ylims[0], ylims[1])
    if multiple_lines == True:
        n = len(ydata)
        if colors == None:
            colors = ['teal', 'darkorange', 'forestgreen', 'darkred', 'black', 'darkviolet', 
                      'saddlebrown', 'royalblue', 'crimson', 'slategrey', 
                      'sienna']
        for i in range(n):
            ax.plot(xdata, ydata[i], label = label[i], color = colors[i], linewidth = linewidth)
    else:
        ax.plot(xdata, ydata, label = label, linewidth = linewidth)
    if label != None:
        plt.legend(frameon=False, loc = legendloc, fontsize = fntsize * legend_size_factor)
    plt.title(title)
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    # increase tick width
    ax.tick_params(width=2)
    ax.set_box_aspect(1)
    plt.yticks(fontsize = fntsize * ticks_size_factor)
    plt.xticks(fontsize = fntsize * ticks_size_factor)
    
def IsoSurface(A, X, Y, Z,
               iso_min = None, iso_max = None,
               iso_surf_n = 3,
               color_scale = 'RdBu_r', opacity = 0.6,
               x_show_caps = False, y_show_caps = False, z_show_caps = False,
               where_to_plot = "browser"):
    
    pio.renderers.default = where_to_plot
    
    fig = go.Figure(data=go.Isosurface(
        x = X.flatten(),
        y = Y.flatten(),
        z = Z.flatten(),
        value = A.flatten(),
        isomin = iso_min,
        isomax = iso_max,
        opacity = opacity,
        colorscale = color_scale,
        surface_count = iso_surf_n,
        colorbar_nticks = iso_surf_n,
        caps=dict(x_show=x_show_caps, y_show=y_show_caps, z_show=z_show_caps)
        ))
    fig.show()
    

def plotly_animate_3d(A, X, Y, Z,
                      iso_min = None, iso_max = None,
                      iso_surf_n = 10, c_min = 0, c_max = 1,
                      color_scale = 'RdBu_r', opacity = 0.6,
                      x_show_caps = False, y_show_caps = False, z_show_caps = False,
                      title = None):
    """Returns fig object that can be animated via fig.show().
    A is array with 4 axis (the first one must be "time" axis).
    X,Y,Z are space meshgrids.
    """
    points = A.shape[0] # number of frames
    frames = [] # we will here store animation frames
    for t in range(points): # iterating through frames
        frames.append(go.Frame(data=go.Isosurface(
            x = X.flatten(),
            y = Y.flatten(),
            z = Z.flatten(),
            value = A[t].flatten(),
            isomin = iso_min,
            isomax = iso_max,
            opacity = opacity,
            colorscale = color_scale,
            surface_count = iso_surf_n,
            colorbar_nticks = iso_surf_n,
            caps=dict(x_show=x_show_caps, y_show=y_show_caps, z_show=z_show_caps),
            cmin = c_min,
            cmax = c_max
            )))
        
    fig = go.Figure(
            data=go.Isosurface(
            x = X.flatten(),
            y = Y.flatten(),
            z = Z.flatten(),
            value = A[0].flatten(),
            isomin = iso_min,
            isomax = iso_max,
            opacity = opacity,
            colorscale = color_scale,
            surface_count = iso_surf_n,
            colorbar_nticks = iso_surf_n,
            caps=dict(x_show=x_show_caps, y_show=y_show_caps, z_show=z_show_caps),
            cmin = c_min,
            cmax = c_max
            ),
            layout=go.Layout(title=title,
                             updatemenus=[dict(type="buttons",
                                              buttons=[dict(label="Play",
                                                            method="animate",
                                                            args=[None
                                                                  ])])]),
            frames=frames)
    
    # fig.layout.updatemenus[0].buttons[0].args[0]['frame']['duration'] = 30
    # fig.layout.updatemenus[0].buttons[0].args[0]['transition']['duration'] = 5


    return fig
    
def plot_3D(X, Y, Z, 
            ax,
            title = "plot_tile", 
            cmap = BuRd):
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap = cmap, edgecolor = 'none')
    ax.set_title(title);
    

