import sys
import configargparse

def get_parser():
    parser = configargparse.ArgumentParser(
        description="Seg Model",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--config', is_config_file=True, help='config file path')
    parser.add_argument("--train_ark", type=str, help="train ark dir")
    parser.add_argument("--train_scp", type=str, help="train scp file")
    parser.add_argument("--train_text", type=str, help="train text file")
    parser.add_argument("--train_len", type=str, help="train feature length file")
    parser.add_argument("--dev_ark", type=str, help="dev ark dir")
    parser.add_argument("--dev_scp", type=str, help="dev scp file")
    parser.add_argument("--dev_text", type=str, help="dev text file")
    parser.add_argument("--dev_len", type=str, help="dev feature length file")
    parser.add_argument("--wordlist", type=str, help="wordlist file")
    parser.add_argument("--data_sample_rate", type=int, default=2, help="data sample rate")
    parser.add_argument("--lstm_sample_rate", type=int, default=4, help="lstm sample rate")
    parser.add_argument("--pooling", type=str, default="concat", help="pooling")
    parser.add_argument("--segment_size", type=int, default=32, help="segment size")
    parser.add_argument("--segment_ratio", type=int, default=2, help="batch segment size/(ilen/olen), training only")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--dev_batch_size", type=int, default=8, help="dev batch size")
    parser.add_argument("--max_ilen", type=int, nargs='+', default=[1000, 1500, 5000], help="input length milestones")
    parser.add_argument("--max_olen", type=int, nargs='+', default=[100, 150, 500], help="output length milestones")
    parser.add_argument("--batch_reduce_ratio", type=float, default=0.5, help="batch size reduction ratio per milestone")
    parser.add_argument("--n_hidden", type=int, default=256, help="hidden dim")
    parser.add_argument("--n_layers", type=int, default=4, help="number of lstm layers")
    parser.add_argument("--log_interval", type=int, default=1000, help="logging interval")
    parser.add_argument("--output", type=str, help="output dir")
    parser.add_argument("--load_awe", type=str, default=None, help="pretrained awe model path")
    parser.add_argument("--load_emb", type=str, default=None, help="pretrained embedding path")
    parser.add_argument("--optim", type=str, default='Adam', help="optimizer")
    parser.add_argument("--scheduler", type=str, default='StepLR', help="scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="scheduler")
    parser.add_argument("--step_size", type=int, default=10, help="scheduler step size")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--nesterov", type=int, default=0, help="nesterov")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--epoch", type=int, default=30, help="number of epochs")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout between LSTM layers")
    parser.add_argument("--max_grad", type=float, default=25, help="threshold for gradient clipping")
    parser.add_argument("--shuffle", action="store_true", help="shuffle training set")
    parser.add_argument("--accum_batch_size", type=int, default=16, help="gradient accumulation batch size")
    parser.add_argument("--lambda_emb", type=float, default=0, help="lambda on regularizing embedding matrix")
    parser.add_argument("--penalize_emb", type=str, default='batch', help="penalization on embedding matrix")
    parser.add_argument("--word_bias", type=int, default=0, help="bias in embedding layer")
    parser.add_argument("--seg_word_samples", type=int, default=2000, help="number of words for segmental loss (training)")

    parser.add_argument("--add_spec_aug", type=int, default=0, help="add spec aug")
    parser.add_argument("--min_flen_aug", type=int, default=10, help="minimum length for spec aug")
    parser.add_argument("--spec_T", type=int, default=40, help="T for spec-aug")
    parser.add_argument("--spec_F", type=int, default=20, help="F for spec-aug")

    parser.add_argument("--amp", type=int, default=1, help="amp training")

    parser.add_argument("--batch_sampler", type=str, default='utt', help="type of batch sampler")
    return parser
