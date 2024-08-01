#%%
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt = '%m%d-%H:%M')

import mne
from mne_bids import read_raw_bids
from pyprep.find_noisy_channels import NoisyChannels
from mne_icalabel import label_components
from scipy.io import loadmat
#%%
def maxwell_filter(raw,ref_head):
    raw_sss = mne.preprocessing.maxwell_filter(raw, 
                                               origin=(0., 0., 0.04),
                                               coord_frame='head',
                                               destination=ref_head,
                                               )
    return raw_sss

def baseline_correction(epochs):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(),times=epochs.times,baseline=(None,0),mode='zscore',copy=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin,event_id=epochs.event_id, metadata=epochs.metadata)
    return epochs

class MEGPreprocessing:
    def __init__(self, bids_path):
        self.bids_path = bids_path
        self.process_info = {}
        self.extra_info   = {}

        self.raw = read_raw_bids(self.bids_path)
        self.raw.load_data()
        self.process_info['runinfo'] = self.bids_path.basename
        
        
        logging.info(f'preprocessing - {self.process_info["runinfo"]}')
        

    def de_linenoise(self, fline = 50):
        raw = self.raw.copy()
        freqs = np.arange(50, 251, 50)
        #TODO: zapline denoise
        
    def fix_bad_channels(self, is_interpolate=True):

        raw = self.raw.copy() 
        if raw.compensation_grade != 0:
            raw.apply_gradient_compensation(0)
            
        auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
            raw=raw, 
            return_scores=True, 
            origin=(0,0,0.04),
            cross_talk=None, 
            calibration=None,
            verbose=True)
        
        if not auto_noisy_chs and not auto_flat_chs:
            self.extra_info['bad_chs'] = 'None'
        else:
            self.extra_info['bad_chs'] = {'noisy': auto_noisy_chs, 'flat': auto_flat_chs, 'scores': auto_scores}
            self.extra_info['N_bad_chs'] = len(auto_noisy_chs+auto_flat_chs)
            
            raw.info['bads'].extend(auto_noisy_chs+auto_flat_chs)
            raw.interpolate_bads()
            raw.apply_gradient_compensation(3)
            self.raw = raw
            
        self.process_info['bad channels'] = 'DONE'
        del raw
        

    def ica_automark(self, save = True, ica_dir=None):
        
        
        raw_resmpl = self.raw.copy().pick_types(meg=True).load_data()
        raw_resmpl.resample(200)
        raw_resmpl.filter(1, 40) # 1Hz以下低频飘逸将很大影响ICA的效果
        

        ica = mne.preprocessing.ICA(method='infomax',random_state=97)
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
        
    
    def mypipeline(self,ica_dir):
        """
        Input: mne.raw
        
        1. bad channel fixing (maxwell)
        2. denoise-line noise
        2.1 denoise-3 gradient compensation
        4. ICA
        
        Out put: mne.raw
        """
        self.fix_bad_channels()
        self.ica_automark(ica_dir=ica_dir)
        # self.ica_plot(ica_dir=ica_dir)
        self.ica_reconst(exclude=None)
        
        annot = self.raw.annotations
        sum_annot = {an: int(np.sum(annot.description == an)) for an in np.unique(annot.description)}
        self.extra_info['annot'] = sum_annot
                
        return self.raw

class MEGEpoching:
    def __init__(self, sub, rawroot, behavdir, metadata_path):
        self.process_info = {}
        self.extra_info   = {}
        self.rawps = sorted([f'{rawroot}/{file}' for file in os.listdir(rawroot) if file.startswith(f'sub-{sub}') and file.endswith('.fif')])
        self.bhvps = sorted([f'{behavdir}/{op.basename(file).split(".")[0]}_meg.mat' for file in self.rawps])
        self.stim_info = pd.read_csv(metadata_path)
        self.basicinfo = {'sub': sub}
        
        self.process_info['runsinfo']   = f'{sub} - {len(self.rawps)} runs'
        logging.info(f'epoching - sub{sub}')
        
    def load_data(self):
        self.raws = [mne.io.read_raw_fif(raw) for raw in self.rawps]
        self.bhvs = [loadmat(bhv) for bhv in self.bhvps]
        self.dev_head_t = raws[0].info['dev_head_t']
        
        # make raws alignable for concatenation
        for raw in self.raws:
             raw = maxwell_filter(raw,ref_head = self.dev_head_t)
            
    def epoching(self,epoch_window=(-0.1,0,8)):
        sub_epo = []
        for i, (raw,bhv) in enumerate(zip(self.raws, self.bhvs)):
            events, event_id = mne.events_from_annotations(raw)
            
            imgTrial = [bhv['runStim'][i][0][0].split('.')[0] for i in range(bhv['runStim'].size)]
            subTrial = []
            
            for img in imgTrial:
                sub_class = stim_info[stim_info['image_id'] == img]['class'].values[0]
                subTrial.append(sub_class)
                    
            supTrial = [stim_info[stim_info['image_id'] == i]['super_class'].values[0] for i in imgTrial]
            ani_resp = np.array(['True' if i == 1 else 'False' for i in bhv['trial'][:,4]])
            ani_labe = np.array(['True' if i == 1 else 'False' for i in bhv['animate_label'][0]])
            judge_acur = [f'{i}' for i in ani_resp == ani_labe]
            rt       = bhv['trial'][:,-1]

            metadata, new_events, new_event_id = mne.epochs.make_metadata(
                events      = events[events[:,2] == event_id['stim_on']],
                event_id    = {'stim_on': event_id['stim_on']},
                tmin        = epoch_window[0],
                tmax        = epoch_window[1],
                sfreq       = raw.info['sfreq']
                )
            stim_ind        = metadata.index[metadata['event_name'] == 'stim_on'].tolist()
            
            fn                                          = self.rawps[i]
            sub                                         = fn.split('/')[-1].split('_')[0][4:]
            ses                                         = fn.split('/')[-1].split('_')[1].split('-')[1]
            run                                         = fn.split('/')[-1].split('_')[2][4:6]
            
            metadata['task']                            = ['ImageNet'] * len(metadata)
            metadata['subject']                         = [sub] * len(metadata)
            metadata['session']                         = [f"{ses}"] * len(metadata)
            metadata['run']                             = [run] * len(metadata)
            metadata.loc[stim_ind, 'image_id']          = imgTrial
            metadata.loc[stim_ind, 'class']             = subTrial
            metadata.loc[stim_ind, 'super_class']       = supTrial
            metadata.loc[stim_ind, 'stim_is_animate']   = ani_labe
            metadata.loc[stim_ind, 'resp_is_animate']   = ani_resp
            metadata.loc[stim_ind, 'resp_is_right']     = judge_acur
            metadata.loc[stim_ind, 'RT']                = rt
            
            epochs = mne.Epochs(
                        raw         = raw,
                        events      = new_events,
                        event_id    = new_event_id,
                        tmin        = epoch_window[0],
                        tmax        = epoch_window[1],
                        metadata    = metadata,
                        # baseline    = (None, 0),
                        picks       = 'meg'
                        )
            epochs_corrected = baseline_correction(epochs)
            sub_epo.append(epochs_corrected)

        self.epoched = mne.concatenate_epochs(epochs_list=sub_epo, add_offset=True)
        self.process_info[sub] = 'DONE'
        self.sub_metadata = metadata
        
    def mypipeline(self):
        self.load_data()
        self.epoching()
        return self.epoched, self.sub_metadata
    