#! /usr/bin/env python3

import mdtraj as md
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from subprocess import run
from operator import sub

k_b = 1.38064852e-23
N_0 = 6.02214076e+23
T = 300


def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled


def get_interval(p, x, alpha):
    '''
    Get an interval in x whose probability p sums to approximately alpha

    Interval is centered on the global maximum of p
    '''
    int_min, int_max = [np.argmax(p)] * 2
    step = "min"
    while True:
        if step == "min":
            int_min = max([int_min - 1, 0])
            step = "max"
        else:
            int_max = min(int_max + 1, len(p)-1)
            step = "min"

        interval_sum = np.sum(p[int_min:int_max + 1])

        if interval_sum >= alpha:
            break
        if int_min == 0 and int_max == len(p)-1:
            raise ValueError('Interval does not fit in array')
    return [x[int_min], x[int_max]]

def histogram(x, bins):
    '''
    Get probability and free energy landscapes for x via a simple histogram

    # Returns
    x : `[float]`
        x coordinates for plotting
    p : `[float]`
        Probability density
    e : `[float]`
        Free energy
    '''
    likelihood, bins = np.histogram(x, bins=bins)
    partition_function = np.sum(likelihood)
    p = likelihood/partition_function
    x = bins[:-1] + 0.5*(bins[:-1] - bins[1:])

    e =  - k_b * T * (np.log(p)) * N_0 / 1000

    return x, p, e

def prob_fes(
        traj,
        colvar=lambda traj: md.rmsd(traj, traj),
        bins=100,
        x_label='Folding coordinate',
        linewidth=2,
        alpha_inner=None,
        alpha_outer=None,
        width=6.5,
        height_ratio=2.5,
        zero=0
    ):
    '''
    Plot probability density alongside the corresponding FES

    # Parameters
    traj : `mdtraj.Trajectory`
        The trajectory to plot
    colvar : Callable: `traj` -> `[float]`
        Function to calculate the folding coordinate from traj
    bins : `int`
        Number of bins to use in the histogram
    x_label : `str`
        Label for the x axes
    linewidth : scalar
        Linewidth for both plots
    alpha_inner, alpha_outer : scalar or `None`
        Interval width in probability around global energy minimum to highlight
    width : scalar
        Width of figure in inches
    height_ratio : scalar
        Ratio width:height of figure
    zero : Scalar
        Value of energy to set as zero

    # Returns
    fig : `matplotlib.figure.Figure`
        The generated figure
    '''
    x = colvar(traj)
    x, p, e = histogram(x, bins=bins)

    fig = plt.figure()  # a new figure window
    ax0 = fig.add_subplot(1, 2, 1)  # specify (nrows, ncols, axnum)
    ax1 = fig.add_subplot(1, 2, 2)  # specify (nrows, ncols, axnum)

    e_plot = e - zero
    start = 0
    stop = list(e_plot).index(np.inf) - 1

    ax0.plot(x, p, linewidth=linewidth)
    ax1.plot(x[start:stop], e_plot[start:stop], linewidth=linewidth)

    ax0.set_ylabel('Probability density, $P$')
    ax0.set_xlabel(x_label)

    ax1.set_ylabel('Free Energy, $\\epsilon_\\mathrm{rel}$ (kJ/mol)')
    ax1.set_xlabel(x_label)

    if alpha_inner is not None:
        inner_min, inner_max = get_interval(p, x, alpha_inner)
        inner_label = f'{alpha_inner*100:.0f}% prob. interval'
        ax0.axvspan(inner_min, inner_max, alpha=0.3, label=inner_label)
        ax1.axvspan(inner_min, inner_max, alpha=0.3, label=inner_label)


    if alpha_outer is not None:
        outer_min, outer_max = get_interval(p, x, alpha_outer)
        outer_label = f'{alpha_outer*100:.0f}% prob. interval'

        ax0.axvspan(outer_min, inner_min, alpha=0.1)
        ax0.axvspan(inner_max, outer_max, alpha=0.1, label=outer_label)

        ax1.axvspan(outer_min, inner_min, alpha=0.1)
        ax1.axvspan(inner_max, outer_max, alpha=0.1, label=outer_label)

    # ax0.legend(frameon=False, borderaxespad=0.0, loc='upper right')
    ax1.legend(frameon=False, borderaxespad=0.0, loc='lower right')

    xmin = x.min()
    xmax = x.max()
    ax0.set_xlim((xmin, xmax))
    ax1.set_xlim((xmin, xmax))

    emin, _ = ax1.get_ylim()
    emax = e_plot[start:stop].max()
    ax1.set_ylim((emin, emax))

    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    height = width / height_ratio
    fig.set_size_inches(width, height)
    fig.tight_layout(pad=0, w_pad=1.0)

    return fig


def cbar_formatter(x, pos):
    '''
    Formatting function for the colorbar with units of kJ/mol
    '''
    return f'${x:0.1f}$ kJ/mol'


def axes_to_data(ax, *args):
    '''
    Transform coordinates from axes space to data space
    '''
    display = ax.transAxes.transform(args)
    inv = ax.transData.inverted()
    return inv.transform(display)


def render(frame, aspectratio=None, cmds=None):
    '''
    Render the frame to a Numpy array with VMD

    # Parameters
    frame : `mdtraj.Trajectory`
        The frame to render. Must be a single frame
    aspectratio : scalar
        Aspect ratio (width/height) of the image
    cmds : `[str]`
        Commands to send to VMD immediately before rendering

    '''
    if len(frame) != 1:
        raise ValueError('Can only render a single frame')
    frame.save_pdb(f'render.pdb')
    if cmds is None:
        cmds = []
    with open('render.vmd', 'w') as f:
        if aspectratio:
            h = 1000
            w = h * aspectratio
            print('display resize', int(round(h)), int(round(w)), file=f)
        print(
            'axes location off',
            'stage location off',
            'display projection orthographic',
            'display rendermode GLSL',
            'color Display Background white',
            'color change rgb 2   0.25   0.25   0.23    ;# gray',
            'mol new render.pdb ;# 0',
            'mol delrep 0 0',
            'mol color ColorID 2',
            'mol material AOChalky',
            'mol representation newcartoon',
            'mol selection protein',
            'mol addrep 0',
            *cmds,
            'render snapshot render.ppm',
            'quit',
            file=f,
            sep='\n'
        )

    run('vmd -startup render.vmd'.split())

    img = mpl.image.imread('render.ppm')
    return img


def pad_hist_range(histogram, padding):
    '''
    Get a range that will pad histogram
    '''
    values_range = np.abs(histogram.min() - histogram.max())
    padding_values = padding * values_range
    return [histogram.min() - padding_values, histogram.max() + padding_values]


def position_from_str(s, width, height, padding):
    '''
    Convert a string to a position for a box of the given size, with padding

    Understands 'upper right', 'upper center', 'upper left', 'center right',
    'center center', 'center left', 'lower right' etc.
    '''
    y, x = s.split()
    if x == 'right':
        x = 1-padding-width
    elif x == 'left':
        x = padding
    elif x == 'center':
        x = 0.5 - (padding + width)/2
    else:
        raise ValueError(f'pos {pos} not recognised')
    if y == 'upper':
        y = 1-padding-height
    elif y == 'lower':
        y = padding
    elif y == 'center':
        y = 0.5 - (padding + height)/2
    else:
        raise ValueError(f'pos {pos} not recognised')

    return (x, y)


def nan_is_white(cmap):
    cmap = mpl.cm.get_cmap(cmap)
    cmap.set_bad('white')
    return cmap


class Fes2d(object):
    '''
    A 2D Free Energy Surface plot of a trajectory

    # Parameters
    traj : `mdtraj.Trajectory`
        The trajectory to compute the FES for
    bins : `int`
        Number of bins to use in the histogram
    x_func, y_func : callable
        Compute the collective variable for each axis from the trajectory
    x_label, y_label : `str`
        Labels of each axis
    width : scalar
        Width of figure in inches
    aspectratio : scalar
        Aspect ratio (w/h) of figure
    zero : scalar
        Value of free energy to set to zero
    padding : scalar
        Padding for laying out figure

    # Attributes
    fig : `matplotlib.figure.Figure`
        The underlying figure of the plot
    ax : `matplotlib.axes.Axes`
        The axes the histogram itself is plotted on. Other axes are
        used simply for layout and are inaccessible as attributes
    traj : `mdtraj.Trajectory`
        The trajectory from which the FES was computed
    xs, ys : `numpy.ndarray`
        Values of collective variables computed from the trajectory
    callout_width, callout_height : scalar
        Width and height of callouts. Set to None unless computed, but
        can be set manually.
    render_cmds : `[str]`
        Commands to send to VMD before rendering frames as images
    '''
    def __init__(
            self,
            traj,
            bins=100,
            x_func=lambda traj: md.rmsd(traj, traj),
            x_label='Folding coordinate',
            y_func=md.compute_rg,
            y_label='Radius of gyration',
            width=6.5,
            aspectratio=1,
            zero=0,
            padding=0.02
        ):
        self.fig = plt.figure()  # a new figure window
        self.ax = self.fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

        self.xs = x_func(traj)
        self.ys = y_func(traj)
        self.traj = traj

        self._padding = padding

        hist, xedges, yedges = np.histogram2d(
            self.xs,
            self.ys,
            bins=75,
            range=[
                pad_hist_range(self.xs, 2 * self._padding),
                pad_hist_range(self.ys, 2 * self._padding)
            ]
        )

        hist_p = hist.T / np.sum(hist)

        eps = - k_b * T * (np.log(hist_p)) * N_0 / 1000 - zero

        self._mesh = self.ax.pcolormesh(
            xedges,
            yedges,
            eps,
            cmap=nan_is_white(mpl.rcParams["image.cmap"]),
            linewidth=0.5,
            edgecolor='face'
        )

        height = width / aspectratio
        self.fig.set_size_inches(width, height)
        self.fig.tight_layout(pad=0.2)

        self.callout_width = None
        self.callout_height = None

        self.savefig = self.fig.savefig
        self.render_cmds = []


    def get_callout_height(self, height, guess):
        '''
        Decide what the height of a callout should be

        # Parameters
        height : scalar
            The height given as an argument
        guess : scalar
            The height that should be used as a default
        '''
        # argument overrides self.callout_height overrides guess
        if height is None and self.callout_height is None:
            height = guess
            self.callout_height = height
        elif self.callout_height is None:
            self.callout_height = height
        elif height is None:
            height = self.callout_height

        return height

    def get_callout_width(self, width, guess):
        '''
        Decide what the width of a callout should be

        # Parameters
        width : scalar
            The width given as an argument
        guess : scalar
            The width that should be used as a default
        '''
        # argument overrides self.callout_width overrides guess
        if width is None and self.callout_width is None:
            width = guess
            self.callout_width = width
        elif self.callout_width is None:
            self.callout_width = width
        elif width is None:
            width = self.callout_width

        return width


    def callout(
        self,
        point,
        box,
        width=None,
        height=None,
        padding=None,
        line_color='0.5',
        line_width=1
    ):
        '''
        Render a callout onto the FES

        The callout is a graphic of the structure in `self.traj` closest
        to `point`. The graphic is placed at `box`, and the point
        corresponding to the chosen structure is labelled with a line to
        to the center of the graphic.

        # Parameters
        point : (scalar, scalar)
            The point to be labelled, in data coordinates (the point in
            collective variable space). The actual coordinates used will
            be slightly different depending on what frames are available.
        box : (scalar, scalar) or `str`
            The coordinates of the bottom left corner of the callout, in
            parent axis coordinates, or a position string
        width, height : scalars
            Width and height of the callout graphic
        padding : scalar
            Amount of padding to use for automatic placement
        line_color : color
            Colour for the line and graphic frame
        line_width : scalar
            Width for the line and graphic frame
        '''
        if padding is None:
            padding = self._padding

        x, y = point

        # argument overrides self.callout_width/height overrides guess
        height = self.get_callout_height(height, 0.15)
        width = self.get_callout_width(width, 0.15)

        if isinstance(box, str):
            x_box, y_box = position_from_str(box, width, height, padding)
        else:
            x_box, y_box = box

        ax2 = self.fig.add_axes([0, 0, 1, 1], label=str(np.random.random()))

        # Place the inset
        ip = InsetPosition(self.ax, [
            x_box, y_box,
            width, height
        ])
        ax2.set_axes_locator(ip)

        # Turn off ticks
        ax2.set_yticks([])
        ax2.set_xticks([])

        x_c, y_c = axes_to_data(self.ax, x_box + width/2, y_box + height/2)

        x_true, y_true = self.display_fitting_structure(ax2, x, y)

        # Draw lines from the point to the center
        line = mpl.lines.Line2D(
            [x_true, x_c],
            [y_true, y_c],
            transform=self.ax.transData,
            color=line_color,
            linewidth=line_width

        )
        self.ax.add_line(line)
        [v.set_color(line_color) for v in ax2.spines.values()]
        [v.set_linewidth(line_width) for v in ax2.spines.values()]


    def distribute_callouts(
        self,
        callouts,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        padding=None,
        width=None,
        height=None
    ):
        '''
        Distribute a series of callouts evenly across the histogram.

        # Parameters
        callouts : `[(float, float)]`
            Data coordinates of the callouts to place (in collective
            variable space). The actual coordinates used will be slightly
            different depending on what frames are available.
        x_min, x_max, y_min, y_max : scalars
            Minimum and maximum coordinates to distribute callouts over.
            Values that aren't specified will be at the edge of the
            histogram (with the specified padding)
        padding : scalar
            Padding to use for laying out callouts. Inherited from the
            object if not specified
        width, height : scalar
            Width and height of the callout graphic boxes
        '''
        if padding is None:
            padding = self._padding

        if x_min is None:
            x_min = padding
        if x_max is None:
            x_max = 1 - padding
        if y_min is None:
            y_min = padding
        if y_max is None:
            y_max = 1 - padding

        n_callouts = len([c for c in callouts if len(c) == 2])

        callout_sep = (1 - 1/n_callouts) * padding


        # argument overrides self.callout_width/height overrides guess
        height = self.get_callout_height(height, (y_max - y_min)/n_callouts)
        width = self.get_callout_width(width, (x_max - x_min)/n_callouts - callout_sep)

        # Distribute the left edges of the panels above the FES
        xs_distrib = np.linspace(
            x_min,
            x_max - width,
            n_callouts
        )
        ys_distrib = np.linspace(
            y_min,
            y_max - height,
            n_callouts
        )
        distrib = zip(xs_distrib, ys_distrib)

        kwargs = dict(width=width, height=height)
        for callout, box in zip(callouts, distrib):
            self.callout(callout, box, **kwargs)


    def inset_colorbar(self, width=0.025, height=0.25, pos='lower right'):
        '''
        Place an inset colorbar within the histogram axes

        # Parameters
        width, height : scalars
            Width and height of the bar itself, excluding labels
        pos : `(x, y)` or `str`
            Where to place the colorbar, as a string or in axes coordinates
        '''
        cax = self.fig.add_axes([0,0,1,1], label=str(np.random.random()))

        try:
            x, y = pos
            x, y = float(x), float(y)
        except ValueError:
            x, y = position_from_str(pos, width, height, self._padding)

        # Place the inset
        ip = InsetPosition(self.ax, [
            x, y,
            width, height
        ])
        cax.set_axes_locator(ip)

        clim = [round(f) for f in self._mesh.get_clim()]
        self._mesh.set_clim(*clim)

        cbar = self.fig.colorbar(
            self._mesh,
            cax=cax,
            format=mpl.ticker.FuncFormatter(cbar_formatter),
            ticks=clim
        )
        cbar.set_label(
            '$\\epsilon_\\mathrm{rel}$',
            rotation='horizontal',
            labelpad=-63,
            horizontalalignment='left',
            verticalalignment='center_baseline',
            rotation_mode='default'
        )

        if 'right' in pos:
            cax.yaxis.set_ticks_position('left')
        else:
            cax.yaxis.set_ticks_position('right')

        return cbar


    def display_fitting_structure(self, axes, x, y):
        '''
        Render the frame in `self.traj` closest to (`x`, `y`) into `axes`

        Returns the actual x and y values of the frame used.
        '''
        dists = np.asarray([self.xs - x, self.ys - y]).T
        frame = np.linalg.norm(dists, axis=-1).argmin()
        l, b, w, h = axes.get_axes_locator().lbwh
        t, r = b + h, l + w
        l, b, t, r = axes.transData.transform([l, b, t, r])
        h, w = t - b, r - l
        print('Targeted', (x, y), 'found', (self.xs[frame], self.ys[frame]))
        img = render(self.traj[frame], aspectratio=(w/h), cmds=self.render_cmds)
        axes.imshow(img)
        return (self.xs[frame], self.ys[frame])


def configure_mpl():
    '''
    Configure MatPlotLib with some defaults
    '''
    mpl.rcdefaults()

    font = {
        'family': 'sans',
        'serif': 'Palatino',
        'size': 11
    }

    plt.rc('font', **font)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('axes', labelsize=11)
    plt.rc('legend', fontsize=11)
    plt.rc('svg', hashsalt="Gimme deterministic SVG output in Josh's thesis")

    mpl.rcParams['axes.prop_cycle'] = cycler(color=[
        '#29549E',
        '#E89005',
        '#3B9A17',
        '#FF0000'
    ])




def folding_coord(traj):
    '''
    A folding coordinate for Chignolin
    '''
    asp3N = traj.top.select('resname ASP and residue 3 and name N')[0]
    asp3O = traj.top.select('resname ASP and residue 3 and name O')[0]
    gly7N = traj.top.select('resname GLY and residue 7 and name N')[0]
    gly7O = traj.top.select('resname GLY and residue 7 and name O')[0]
    thr8O = traj.top.select('resname THR and residue 8 and name O')[0]

    atom_pairs = [[asp3N, gly7O], [asp3O, gly7N], [asp3N, thr8O]]

    labels = ['Asp3N-Gly7O', 'Asp3O-Gly7N', 'Asp3N-Thr8O']
    dists = md.compute_distances(traj, atom_pairs=atom_pairs)

    return np.sum(dists - np.array([0.6, 0.3, 0.3]), axis=1) / 3


if __name__ == '__main__':
    traj = md.load("cln025_c22s.pdb")
    traj = md.load("cln025_c22s.xtc", top=traj.top)

    configure_mpl()

    start_ns = 400
    start = int(start_ns * 1000 / traj.timestep)

    zero_of_energy = 15 # kj/mol

    fig = prob_fes(
        traj[start::],
        colvar=folding_coord,
        bins=250,
        x_label='Folding coordinate, $x$ (nm)',
        alpha_inner=0.78,
        zero=zero_of_energy
    )
    fig.savefig('../cln025_prob_fes.svg', dpi=600)

    fes = Fes2d(
        traj[start::],
        bins=75,
        x_func=folding_coord,
        x_label='Folding coordinate, $x$ (nm)',
        y_func=md.compute_rg,
        y_label='Radius of gyration, $R_g$ (nm)',
        zero=zero_of_energy
    )
    # Choose a nice rendering orientation for the callouts
    fes.render_cmds = [
        'scale by 1.75',
        'rotate y by 100',
        'rotate z by 135',
        'rotate x by 30'
    ]
    # Distribute 5 callouts on a diagonal above the surface
    fes.distribute_callouts([
        (0.00, 0.6),  # Folded
        (0.26, 0.59),  # Misfolded
        (0.45, 0.775),  # Disordered intermediate
        (0.79, 0.805),  # Disordered intermediate
        (1.02, 1.08)  # Disordered edge
    ], x_max=0.75, y_min=0.30)
    # Place another callout in the empty bottom right corner
    fes.callout((0.76, 0.62), 'lower right')  # Corner
    # Place the colorbar in the empty top left corner
    fes.inset_colorbar(pos='upper left')
    fes.savefig('../cln025_fes2d.svg', dpi=600)

