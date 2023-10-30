import torch
import argparse
from qa_dataset import qa_dataset

tmp_dataset = qa_dataset(data_dict=None)

def generate(model, query_str='A', max_predict_len=100, device=None):
    # edge case, put a default query
    if len(query_str) == 0:
        query_str = 'A'

    # encode query string to char tensor, you can print it out
    query_input = tmp_dataset.get_tensor(query_str).to(device)

    # TODO: Explain, why we need pass [BEG] token into generate
    query_answer_init_token = tmp_dataset.get_beg_tensor().to(device)
    # TODO: Explain function of encoding stage
    predicted = model.generate(query_input, query_answer_init_token,
                               tmp_dataset.pos_end_token, max_predict_len, device=device)

    # if last token is [END], remove it and not print
    if predicted[-1] == tmp_dataset.pos_end_token:
        predicted = predicted[:-1]

    # convert token index back into character
    char_list = [tmp_dataset.get_char(ind) for ind in predicted]
    answer_str = ''.join(char_list)

    return answer_str


# Run as standalone script
if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model_path', type=str, default='saves/rnn.pt')
    argparser.add_argument('-q', '--query_str', type=str, required=True)
    argparser.add_argument('-l', '--predict_len', type=int, default=300)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    # if you have a GPU, turn this on
    use_cuda = args.cuda and torch.cuda.is_available()

    # if you have a MAC M1/2 chip, turn this on
    use_mps = False  # args.mps or torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # print('The device is', device)

    model = torch.load(args.model_path, map_location='cpu').to(device)
    response = generate(model, args.query_str, args.predict_len, device)
    print('Q: %s' % (args.query_str,))
    print('A: %s' % (response,))
