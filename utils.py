import torch
import torch.nn as nn

class EarlyStopper:
    def __init__(self, patience=1000, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Npool(nn.Module):
    def __init__(self, pool_size, stride, n_neighbor):
        super(Npool, self).__init__()
        self.n_neighbor = n_neighbor
        self.pool_size = pool_size
        self.stride = stride

    def get_weight(self, n_neighbor):
        k = 1/(n_neighbor + 1)
        weight = torch.zeros(n_neighbor + 1)
        for e in range(n_neighbor + 1):
            weight[e] = (1 - e * k) * k
        return torch.concat((torch.flip(weight[1:], dims=[-1]), weight))

    def apply_fn(self,input, fn):
        s_list=[]
        for s in  input:
            c_list = []
            for c in s:
                h_list = []
                for h in c:
                    h_list.append(fn(h))
                channel = torch.stack(h_list)
                c_list.append(channel)
            sample = torch.stack(c_list)
            s_list.append(sample)
        output = torch.stack(s_list)

        return output

    def pool1d(self, input):
        assert len(input.shape)==1
        device = input.device
        out_size =(input.shape[0] - self.pool_size)// self.stride + 1 
        out_put = torch.zeros(out_size).to(device)
     
        for i in range(out_size):
            current_win = input[i*self.stride:i*self.stride + self.pool_size]
            max_idx = torch.argmax(current_win)
            
            # handle edge cases when there aren't enough neighbors
            for left in range(self.n_neighbor+1):
                if (max_idx - left<0):
                    left-=1
                    break
            for right in range(self.n_neighbor+1):
                if ((max_idx + right) > (self.pool_size-1)):
                    right-=1
                    break  

            eff_neighbor = min(left, right)
            #print(eff_neighbor)
            weight = self.get_weight(eff_neighbor).to(device)
            max_win = current_win[max_idx - eff_neighbor: max_idx + eff_neighbor+1]

            weighted_max = torch.sum(weight * max_win)
            out_put[i]= weighted_max
   
        return out_put
        
    def forward(self, input):
        return self.apply_fn(input, self.pool1d)