from configparser import ConfigParser
#from dbm.data import *
from dbm.gan_seq_ljrep import *
#from dbm.gan import *
#from dbm.ff import *
#import gan_seq
#import gan
import argparse



def main():

    #torch device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU will be used!!!")
    else:
        device = torch.device("cpu")
        print("CPU will be used!")
    torch.set_num_threads(12)

    # ## Read config

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config_file = args.config

    cfg = ConfigParser()
    cfg.read(config_file)

    #set up model
    model = GAN_SEQ(device=device, cfg=cfg)

    #with open('./' + name + '/config.ini', 'a') as f:
    #    cfg.write(f)

    #train
    model.train()


if __name__ == "__main__":
    main()