#%%
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt = '%m%d-%H:%M')

import mne
from pyprep.find_noisy_channels import NoisyChannels
from mne_icalabel import label_components
from scipy.io import loadmat

#%%
class MEGPreprocessing:
    def __init__(self, subject, task, session, run, bids_dir):
        self.process_info = {}
        self.extra_info   = {}

        self.bids_path = BIDSPath(subject=subject, task=task, root=bids_dir, session=session, run=run)
        self.raw = read_raw_bids(self.bids_path)
        self.raw.load_data()
        self.process_info['runinfo'] = self.bids_path.basename
        
        
        logging.info(f'preprocessing - {self.process_info["runinfo"]}')
        

     
    def find_bad_channels(self, is_interpolate=True):
        raw_nd = self.raw.copy()
        nd = NoisyChannels(raw_nd, random_state=1337)
        nd.find_bad_by_correlation()
        nd.find_bad_by_deviation()
        nd.find_bad_by_ransac()
        
        bads = nd.get_bads()
        raw_nd.info['bads'].extend(bads)
        if is_interpolate:
            raw_nd.interpolate_bads(reset_bads=True)
        
        self.raw = raw_nd
        self.process_info['bad channels fixing'] = 'DONE'
        self.extra_info['bad channels fixing'] = nd.get_bads(as_dict=True)
        # pprint(self.process_info)
        

    def zapline_denoise(self, fline = 50):
        data = self.raw.get_data().T
        data = np.expand_dims(data, axis=2)  # 现在形状为 (nsample, nchan, ntrial(1))
        sfreq = self.raw.info['sfreq']
        with contextlib.redirect_stdout(io.StringIO()): # mute for simplicity
            out, _ = dss.dss_line_iter(data, fline, sfreq, nfft=400)
            cleaned_raw = mne.io.RawArray(out.T.squeeze(), self.raw.info)
            cleaned_raw.set_annotations(self.raw.annotations)
        
        self.raw = cleaned_raw 
        self.process_info['zapline denoise'] = 'DONE'
        self.extra_info['zapline denoise'] = f'fline - {fline}'
        # pprint(self.process_info)

                
    def rereference(self, ref='average'):
        refraw = self.raw.copy()
        refraw.set_eeg_reference(ref)
        self.raw = refraw
        
        if 're_reference bef ICA' in self.process_info:
            self.process_info['re_reference aft ICA'] = 'DONE'
            self.extra_info['re_reference aft ICA'] = ref
            
        self.process_info['re_reference bef ICA'] = 'DONE'
        self.extra_info['re_reference bef ICA'] = ref
        # pprint(self.process_info)        
    
    def ica_automark(self, save = True, ica_dir=None):
        
        
        raw_resmpl = self.raw.copy().pick_types(eeg=True).load_data()
        raw_resmpl.resample(200)
        raw_resmpl.filter(1, 40)
        

        ica = mne.preprocessing.ICA(method='fastica',random_state=97,n_components=40)
        ica.fit(raw_resmpl)
        
        ic_labels = label_components(raw_resmpl, ica, method="iclabel")
        exclude = [i for i,l in enumerate(ic_labels['labels']) if l not in ['brain','other']]
        ica.exclude = exclude
        
        if save:
            fn = self.process_info['runinfo']
            ica_dir = os.path.join(ica_dir, 'icafif')
            os.makedirs(ica_dir, exist_ok=True)
            ica.save(os.path.join(ica_dir, f'{fn}_ica.fif'), overwrite=True)

        self.ica = ica
        self.raw4plot = raw_resmpl
        self.process_info['ICA auto label'] = 'DONE'
        self.extra_info['ICA auto label'] = {i:l for i,l in enumerate(ic_labels['labels'])}
        self.extra_info['ICA auto score'] = {i:float(s) for i,s in enumerate(ic_labels['y_pred_proba'])}
        self.extra_info['ICA Nartifact'] = len(exclude)
        # pprint(self.process_info)
            
    def ica_plot(self, ica_dir=None):
        #set path for figs
        fn = self.process_info['runinfo']
        fig_dir = os.path.join(ica_dir, 'figs')
        ics_fig = os.path.join(fig_dir, 'ICs_all')
        proper_dir = os.path.join(fig_dir, 'IC_properties')
        source_fig = os.path.join(fig_dir, 'IC_sources')
        proper_run_dir = os.path.join(proper_dir, fn)
        for dir in [fig_dir, ics_fig, proper_dir, source_fig, proper_run_dir]:
            os.makedirs(dir, exist_ok=True)

        raw_ica         = self.ica
        raw_resmpl      = self.raw4plot
        labels          = self.extra_info['ICA auto label']
        scores          = self.extra_info['ICA auto score']
        
        # ICs all plot
        # n_components    = ica.n_components_
        # n_rows          = 5
        # n_cols          = 4
        # plt.ioff()
        # fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20))
        # for i in range(n_components):
        #     row = i // n_cols
        #     col = i % n_cols
        #     ax = axes[row, col]
        #     raw_ica.plot_components(picks=i, axes=ax, sphere=(0, 0.02, 0, 0.09), show=False)
        #     ax.set_title(f'{i}-{labels[i]}', fontsize=40)
        # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.6)
        # plt.savefig(os.path.join(ics_fig, f'{fn}.png'))
        # plt.close(fig)
        
        
        n_components    = raw_ica.n_components_
        n_cols = 5
        n_rows = 8
        plt.ioff()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 6))
        for i in range(n_components):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_components > 1 else axes  # 当只有一个组件时，axes不是二维数组
            raw_ica.plot_components(picks=i, axes=ax, sphere=(0, 0.02, 0, 0.09), show=False)
            ax.set_title(f'{i} - {labels[i]}({scores[i]:.2f})', fontsize=40)

        # 移除多余的子图
        # for i in range(n_components, n_rows * n_cols):
        #     fig.delaxes(axes.flatten()[i])

        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.6)
        plt.savefig(os.path.join(ics_fig, f'{fn}.png'))
        plt.close(fig)
        
        # every IC's properties plot
        for comp in range(n_components):
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            with contextlib.redirect_stdout(io.StringIO()):
                raw_ica.plot_properties(raw_resmpl, picks=comp, topomap_args = {'sphere' : (0, 0.02, 0, 0.09)}, show=False, verbose=False)
            plt.savefig(os.path.join(proper_run_dir, f'{comp}.png'))
            plt.close(fig)
        
        # ICA source plot
        raw_ica.plot_sources(raw_resmpl, picks = [i for i in range(0,20)], show=False)
        plt.savefig(os.path.join(source_fig, f'{fn}-0.png'))
        raw_ica.plot_sources(raw_resmpl, picks = [i for i in range(20,40)], show=False)
        plt.savefig(os.path.join(source_fig, f'{fn}-1.png'))
        plt.close('all') 
        plt.ion()  

        self.process_info['ICA plot'] = 'DONE'
        # pprint(self.process_info)

    def ica_reconst(self,exclude):
        
        raw = self.raw.copy()
        raw.load_data()
        raw.resample(200)
        raw.filter(0.1, 40)
        
        if exclude:
            self.ica.exclude = exclude
        
        self.ica.apply(raw)
        self.raw = raw        
        self.process_info['ICA reconst'] = 'DONE'
        self.extra_info['ICA reconst'] = 'bandfilter: 0.1-40\nsplfreq: 200'
        # pprint(self.process_info)
        
    
    def mypipeline(self,ica_dir,exclude):
        """
        Input: mne.raw
        
        1. set montage 1005
        2. bad channel detection (pyprep)
        3. denoise-line noise
        4. denoise-ASR
        4. ICA
        5. re-referencing
        
        Out put: mne.raw
        """
        self.apply_montage()
        self.zapline_denoise()
        self.find_bad_channels()
        
        self.rereference()#like eeglab, re-reference before ICA
        self.ica_automark(ica_dir=ica_dir)
        self.ica_plot(ica_dir=ica_dir)
        self.ica_reconst(exclude=exclude)
        self.rereference() #re-reference after ICA also
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
                
        return self.raw
