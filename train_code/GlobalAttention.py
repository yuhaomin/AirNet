import torch
import torch.nn as nn


class GlobalAttention(nn.Module):

    def __init__(self,args, dim_out):
        super(GlobalAttention, self).__init__()

        # self.dim = dim
        self.hidR = args.hidRNN;
        self.hidRx = args.hidRNNx;
        self.hidRy = args.hidRNNy;

        self.linear_context = nn.Linear(self.hidRx+self.hidR, self.hidRy, bias=False)
        self.linear_query = nn.Linear(self.hidRy, self.hidRy, bias=True)
        self.v = nn.Linear(self.hidRy, 1, bias=False)

        self.linear_out = nn.Linear(self.hidRy +(self.hidRx+self.hidR)* 2, dim_out)

        self.sm = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()


    def score(self, h_t, h_s):

        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()


        wq = self.linear_query(h_t.view(-1, self.hidRy))
        wq = wq.view(tgt_batch, tgt_len, 1, self.hidRy)
        wq = wq.expand(tgt_batch, tgt_len, src_len, self.hidRy)

        uh = self.linear_context(h_s.contiguous().view(-1, self.hidRx+self.hidR))
        uh = uh.view(src_batch, 1, src_len, self.hidRy)
        uh = uh.expand(src_batch, tgt_len, src_len, self.hidRy)

        wquh = self.tanh(wq + uh)

        return self.v(wquh.view(-1, self.hidRy)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input,h1, memory_bank, dim_out):

        input = input.permute(1,0,2)
        h1 = h1.permute(1,0,2)
        batch = memory_bank.shape[1]
        memory_bank = memory_bank.permute(1, 0, 2)

        memory_bank1 = memory_bank.view(batch, 7, 24, self.hidRx + self.hidR)
        memory_bank1 = memory_bank1[:, :, -1:, :].contiguous();
        memory_bank1 = memory_bank1.view(batch, 7, self.hidRx + self.hidR)


        memory_bank_aj = memory_bank[:, -6:, :]

        memory_bank = torch.cat([memory_bank_aj, memory_bank1], 1)

        batch, sourcel, dim = memory_bank.size()
        batch_, targetl, dim_ = input.size()

        align = self.score(input, memory_bank)

        align_vectors = self.sm(align.view(batch * targetl, sourcel))
        align_vectors = align_vectors.view(batch, targetl, sourcel)

        c = torch.bmm(align_vectors, memory_bank)

        concat_c = torch.cat([c, input, h1], 2).view(batch * targetl, self.hidRy + (self.hidRx + self.hidR) * 2)
        attn_h = self.linear_out(concat_c).view(batch, targetl, dim_out)

        return attn_h, align_vectors
