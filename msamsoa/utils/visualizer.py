from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Colorbar Kebutuhan Pupuk
f1 = [219,219,219] #dbdbdb : Butuh Pupuk
f2 = [255,255,255] #ffffff : Tidak Butuh dikunjungi
f_col = np.array([f1, f2], np.uint8)
f_lab = ['Butuh Pupuk', 'Tidak Butuh Pupuk']
f_rcol = f_col.reshape(1,2,3)

f_ncol = np.hstack((np.vectorize(lambda x:x/255)(f_col), np.ones((2,1))))
f_lcm = ListedColormap(f_ncol)
f_bound = [i for i in range(3)]
f_norm = BoundaryNorm(f_bound, f_lcm.N)

# Fungsi
## Membuat colorbar untuk subplot
def generateColorbar(ax, plot, ticks, width="5%", height="100%", loc='lower left', bta=(1.05, 0., 1, 1),
                     bpad=0):
    cbar = inset_axes(ax,
                      width=width,
                      height=height,
                      loc=loc,
                      bbox_to_anchor=bta,
                      bbox_transform=ax.transAxes,
                      borderpad=0,
                     )
    plt.colorbar(plot, cax=cbar, ticks=ticks)

## Menampilkan Lahan
def showScenarios(scenarios, lcm=f_lcm, norm=f_norm):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4*len(scenarios),4))

    if(len(scenarios) == 1):
        plt.setp(axes, xticks=[], yticks=[])
        plot = axes.imshow(scenarios[0], cmap=lcm, norm=norm, vmin=0, vmax=1)
        generateColorbar(axes, plot, [0,1])
    else:
        for (ax, sce) in zip(axes, scenarios):
            plt.setp(ax, xticks=[], yticks=[])
            plot = ax.imshow(sce, cmap=lcm, norm=norm, vmin=0, vmax=1)
            generateColorbar(ax, plot, [0,1])

## Membuat Animasi dengan Peta Pemupukan
def generateSimulationFertilized(s_tracker, z_tracker, iteration=None, lcm=f_lcm, norm=f_norm):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    s_iter = s_tracker.iteration
    z_iter = z_tracker.iteration
    iteration = max(s_iter, z_iter) if (iteration == None) else iteration

    plt.subplots_adjust(wspace=0.3)
    nplots, bplots = ([], [])
    for ax in axes:
        plt.setp(ax, xticks=[], yticks=[])
        nplot, = ax.plot([], [], 'o', color='blue')
        bplot, = ax.plot([], [], 'o', color='red')
        nplots.append(nplot)
        bplots.append(bplot)

    splot = axes[0].imshow(s_tracker.tm_tracks[0], cmap=lcm, norm=norm, vmin=0, vmax=1)
    generateColorbar(axes[0], splot, [0,1])

    zplot = axes[1].imshow(z_tracker.tm_tracks[0], cmap=lcm, norm=norm, vmin=0, vmax=1)
    generateColorbar(axes[1], zplot, [0,1])

    def animate(i):
        print("Working on iteration...{}".format(i))
        its = min(i, s_iter)
        itz = min(i, z_iter)
        (sny, snx), (sby, sbx) = s_tracker.getAgentLocations(its)
        nplots[0].set_data(snx, sny)
        bplots[0].set_data(sbx, sby)

        (zny, znx), (zby, zbx) = z_tracker.getAgentLocations(itz)
        nplots[1].set_data(znx, zny)
        bplots[1].set_data(zbx, zby)

        s_srate = (s_tracker.s_tracks[its]/s_tracker.size)*100
        s_frate = (s_tracker.f_tracks[its]/s_tracker.nt)*100
        axes[0].set_title(("Iterasi: {}\nSurveillance:{:6.2f}%. Pemupukan: {:6.2f}%"
                          ).format(i, s_srate, s_frate), {'fontsize':12}, loc='left')
        z_srate = (z_tracker.s_tracks[itz]/z_tracker.size)*100
        z_frate = (z_tracker.f_tracks[itz]/z_tracker.nt)*100
        axes[1].set_title(("Surveillance:{:6.2f}%. Pemupukan: {:6.2f}%"
                          ).format(z_srate, z_frate), {'fontsize':12}, loc='left')

        splot = axes[0].imshow(s_tracker.tm_tracks[its], cmap=lcm, norm=norm, vmin=0, vmax=1)
        zplot = axes[1].imshow(z_tracker.tm_tracks[itz], cmap=lcm, norm=norm, vmin=0, vmax=1)
        return splot, zplot

    # Menampilkan Animasi
    frames = np.arange(0, iteration, 1)
    anim = FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.close(anim._fig)
    return HTML(anim.to_html5_video())

## Membuat Animasi dengan Peta Kunjungan
def generateSimulationVisited(s_tracker, z_tracker, iteration=None, lcm=f_lcm, norm=f_norm):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    s_iter = s_tracker.iteration
    z_iter = z_tracker.iteration
    iteration = max(s_iter, z_iter) if (iteration == None) else iteration

    plt.subplots_adjust(wspace=0.3)
    nplots, bplots = ([], [])
    for ax in axes:
        plt.setp(ax, xticks=[], yticks=[])
        nplot, = ax.plot([], [], 'o', color='blue')
        bplot, = ax.plot([], [], 'o', color='red')
        nplots.append(nplot)
        bplots.append(bplot)

    splot = axes[0].imshow(s_tracker.vm_tracks[0], cmap=lcm, norm=norm, vmin=0, vmax=1)
    generateColorbar(axes[0], splot, [0,1])

    zplot = axes[1].imshow(z_tracker.vm_tracks[0], cmap=lcm, norm=norm, vmin=0, vmax=1)
    generateColorbar(axes[1], zplot, [0,1])

    def animate(i):
        print("Working on iteration...{}".format(i))
        its = min(i, s_iter)
        itz = min(i, z_iter)
        (sny, snx), (sby, sbx) = s_tracker.getAgentLocations(its)
        nplots[0].set_data(snx, sny)
        bplots[0].set_data(sbx, sby)

        (zny, znx), (zby, zbx) = z_tracker.getAgentLocations(itz)
        nplots[1].set_data(znx, zny)
        bplots[1].set_data(zbx, zby)

        s_srate = (s_tracker.s_tracks[its]/s_tracker.size)*100
        s_frate = (s_tracker.f_tracks[its]/s_tracker.nt)*100
        axes[0].set_title(("Iterasi: {}\nSurveillance:{:6.2f}%. Pemupukan: {:6.2f}%"
                          ).format(i, s_srate, s_frate), {'fontsize':12}, loc='left')
        z_srate = (z_tracker.s_tracks[itz]/z_tracker.size)*100
        z_frate = (z_tracker.f_tracks[itz]/z_tracker.nt)*100
        axes[1].set_title(("Surveillance:{:6.2f}%. Pemupukan: {:6.2f}%"
                          ).format(z_srate, z_frate), {'fontsize':12}, loc='left')

        splot = axes[0].imshow(s_tracker.vm_tracks[its], cmap=lcm, norm=norm, vmin=0, vmax=1)
        zplot = axes[1].imshow(z_tracker.vm_tracks[itz], cmap=lcm, norm=norm, vmin=0, vmax=1)
        return splot, zplot

    # Menampilkan Animasi
    frames = np.arange(0, iteration, 1)
    anim = FuncAnimation(fig, animate, frames=frames, interval=100)
    plt.close(anim._fig)
    return HTML(anim.to_html5_video())
