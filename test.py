import torch
import torch.nn.functional as F

def main():

    highly_pos = 15 * torch.ones(1)
    highly_neg = -15 * torch.ones(1)

    out_high = -1 * F.logsigmoid(highly_pos) 
    out_low = -1 * F.logsigmoid(highly_neg)

    print('Highly Positive: {}; Highly Negative: {}'.format(out_high, out_low))



if __name__ == '__main__':

    m = torch.distributions.exponential.Exponential(torch.tensor([2.75]))
    x = 5
    #main()
