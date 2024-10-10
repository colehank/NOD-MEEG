#%%
import numpy as np
from scipy.spatial import ConvexHull
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import PIL.Image
from scipy.io import savemat
import os
import mne
plt.rcParams['figure.max_open_warning'] = 60
#%%

def cart2sph(x, y, z):
    xy = np.sqrt(x * x + y * y)
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, xy)
    return r, theta, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def cart2pol(x, y):
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return r, theta

def make_head_outlines_new(sphere, pos, outlines, clip_origin):
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius * 1.01 + x
    head_y = np.sin(ll) * radius * 1.01 + y
    dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
    dx, dy = dx.real, dx.imag
    
    outlines_dict = dict(head=(head_x, head_y))
    
    mask_scale = 1.
    mask_scale = max(
        mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius
    )
    
    outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
    clip_radius = radius * mask_scale
    outlines_dict['clip_radius'] = (clip_radius,) * 2
    outlines_dict['clip_origin'] = clip_origin
    
    outlines = outlines_dict
    
    return outlines

def prepare_for_megnet(ica,raw,results_dir):
    os.makedirs(results_dir,exist_ok = True)
    data_picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
        mne.viz.topomap._prepare_topomap_plot(ica, 'mag')

    mags = mne.pick_types(ica.info, meg='mag')
    channel_info = ica.info['chs']

    channel_locations3d = np.zeros([len(mags), 3])
    for i in np.arange(0, len(mags)):
        tmp = channel_info[i]
        channel_locations3d[i, :] = tmp['loc'][0:3]

    channel_locations_3d_spherical = np.transpose(cart2sph(channel_locations3d[:, 0], channel_locations3d[:, 1], channel_locations3d[:, 2]))

    TH = channel_locations_3d_spherical[:, 1]
    PHI = channel_locations_3d_spherical[:, 2]

    channel_locations_2d = np.zeros([len(mags), 2])
    newR = 1 - PHI / np.pi * 2
    channel_locations_2d = np.transpose(pol2cart(newR, TH))
    X = channel_locations_2d[:, 0]
    Y = channel_locations_2d[:, 1]

    hull = ConvexHull(channel_locations_2d)
    Border = hull.vertices
    Dborder = 1 / newR[Border]

    FuncTh = np.hstack([TH[Border] - 2 * np.pi, TH[Border], TH[Border] + 2 * np.pi])
    funcD = np.hstack((Dborder, Dborder, Dborder))
    finterp = interpolate.interp1d(FuncTh, funcD)
    D = finterp(TH)

    newerR = np.zeros((len(mags),))
    for i in np.arange(0, len(mags)):
        newerR[i] = min(newR[i] * D[i], 1)
    [Xnew, Ynew] = pol2cart(newerR, TH)
    pos_new = np.transpose(np.vstack((Xnew, Ynew)))

    outlines_new = make_head_outlines_new(np.array([0, 0, 0, 1]), pos_new, 'head', (0, 0))
    outline_coords = np.array(outlines_new['head'])

    n_components = ica.n_components_
    for comp in np.arange(0, n_components, 1):
        data = np.dot(ica.mixing_matrix_[:, comp].T, ica.pca_components_[:ica.n_components_])
        # set up the figure canvas
        fig = plt.figure(figsize=(1.3, 1.3), dpi=100, facecolor='black')
        canvas=FigureCanvas(fig)
        ax = fig.add_subplot(111)
        mnefig, contour = mne.viz.plot_topomap(data, 
                                            pos_new, 
                                            sensors=False, 
                                            outlines=outlines_new, 
                                            extrapolate='head', 
                                            sphere=[0, 0, 0, 1], 
                                            contours=10, 
                                            res=120, 
                                            axes=ax,
                                            show=False, 
                                            cmap='bwr')
        
        outpng = f'{results_dir}/component{str(int(comp)+1)}.png'
        mnefig.figure.savefig(outpng,dpi=120,bbox_inches='tight',pad_inches=0)
        rgba_image=PIL.Image.open(outpng)
        rgb_image=rgba_image.convert('RGB')
        os.remove(outpng)
        
        # save the RGB image as a .mat file
        mat_fname = f'{results_dir}/component{str(int(comp)+1)}.mat'
        savemat(mat_fname, {'array':np.array(rgb_image)})
        
        del mnefig,fig
        plt.close('all')

    # Save ICA timeseries as input for classification
    # Currently inputs to classification are matlab arrays
    ica_ts = ica.get_sources(raw)._data.T
    outfname = f'{results_dir}/ICATimeSeries.mat' #'{file_base}-ica-ts.mat'
    savemat(outfname, {'arrICATimeSeries':ica_ts})



#%%
if __name__ == '__main__':
    raw = mne.io.read_raw_fif('/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_codes/utils/ICA_megnet/raw.fif')
    raw.pick(['mag'])
    ica=mne.preprocessing.ICA(n_components = 20,max_iter='auto')
    ica.fit(raw)
    prepare_for_megnet(ica,raw,results_dir = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/NOD-MEEG_codes/utils/ICA_megnet')

