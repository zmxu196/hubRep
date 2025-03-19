import logging
import datetime
import os


def get_logger(args,timestamp):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    log_txt_path_dir = 'results/'+str(args.dataset_name)+'/logs/'
    if not os.path.exists(log_txt_path_dir):
        os.makedirs(log_txt_path_dir)
    fh = logging.FileHandler(log_txt_path_dir+str(args.dataset_name)+'_'+str(timestamp) + '.log')   

    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Ignore DEBUG level logs
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

