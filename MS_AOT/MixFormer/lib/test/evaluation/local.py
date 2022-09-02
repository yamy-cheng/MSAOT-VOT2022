from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/got10k_lmdb'
    settings.got10k_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/lasot_lmdb'
    settings.lasot_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/lasot'
    settings.network_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/nfs'
    settings.otb_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/OTB2015'
    settings.prj_dir = '/home/cym/project/vot2022sts/MS_AOT/MixFormer'
    settings.result_plot_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/test/result_plots'
    settings.results_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/cym/project/vot2022sts/MS_AOT/MixFormer'
    settings.segmentation_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/test/segmentation_results'
    settings.tc128_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/trackingNet'
    settings.uav_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/UAV123'
    settings.vot_path = '/home/cym/project/vot2022sts/MS_AOT/MixFormer/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

