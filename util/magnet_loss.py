"""
A PyTorch implmentation of the KL-Divergence Loss as described in (https://arxiv.org/abs/1511.06321)

Lua Implementation (not inspected yet TODO) (https://github.com/yenchanghsu/NNclustering/blob/master/BatchKLDivCriterion.lua)

"""

import torch
from torch import nn


class magnet_loss(nn.Module):
    def __init__(self, D = 12, M = 4, alpha = 7.18):
        super(magnet_loss, self).__init__()

        self.D      = D
        self.M      = M
        self.alpha  = alpha

    def forward(self, outputs, indices, assignment, loss_vector, loss_count, images, model):
        """
        :param  indices     The index of each embedding
        :param  outputs     The set of embeddings
        :param  clusters    Cluster assignments for each index
        :return Loss        Magnet loss calculated for current batch
        """
        _min_float  = 1e-6

        outputs     = outputs.float()
        batch_size  = outputs.size(0)

        loss        = torch.zeros(1)

        # If GPU is available compute loss on it
        loss    = loss.cuda()
        loss    = torch.autograd.Variable(loss).cuda()

        ######################### Cluster Assignments ##########################
        # Generate a set of clusters in the batch
        # and the local indices corresponding to each of those clusters
        # batch_clusters = { cluster_number : [ local_indices] }
        # TODO fix later!!!  -- for now assiming indices are irrelevant!
        batch_clusters = {}
        for i in range(0, len(indices)):
            curr_cluster = assignment[indices[i]]
            if curr_cluster in batch_clusters.keys():
                batch_clusters[curr_cluster].append(i)
            else:
                batch_clusters[curr_cluster] = [i]

        ######################### Cluster Assignments ##########################
        clusters = list(batch_clusters.keys())

        ##################### Calculate Means and STDEV ########################
        num_instances = 0.0
        stdev       = torch.zeros(1)        # sdev array
        stdev       = stdev.cuda()
        stdev       = torch.autograd.Variable(stdev)
        """
        c_means = []

        for i in range(0, outputs.size(0)):
            c_means.append(torch.zeros(outputs.shape[1]))                  # sdev array
            c_means[i]   = c_means[i].cuda()
            c_means[i]   = torch.autograd.Variable(c_means[i])
        """
        c_means = torch.stack([torch.mean(outputs[batch_clusters[clusters[m]]], dim=0) for m in range(0, len(clusters))])
        
        for m in range(0, len(clusters)):
            c = clusters[m]

            for d in range(0, len(batch_clusters[c])):
                stdev += (outputs[batch_clusters[c][d]] -  c_means[m]).norm(p=2).pow(2)
                num_instances += 1.0

        stdev = stdev / (num_instances - 1.0)
        variance = stdev.pow(2)

        ########################## CALCULATE THE LOSS #########################
        denom = []
        for i in range(0, outputs.size(0)):
            denom.append(torch.zeros(1))                  # sdev array
            denom[i]   = denom[i].cuda()
            denom[i]   = torch.autograd.Variable(denom[i])

        for m in range(0, len(clusters)):
            c = clusters[m]
            for d in range(0, len(batch_clusters[c])):
                ind   = batch_clusters[c][d]
                for mF in range(0, len(clusters)):
                    if mF != m:
                        denom[ind] += (-1 * (outputs[batch_clusters[c][d]] - c_means[mF]).norm(p=2).pow(2) / (2 * variance + 1e-8) ).exp() # adding epsilon 1e-8 to denominator to prevent becoming NaN

                denom[ind] += 1e-8
                loss -=  (( ( -1 * (outputs[batch_clusters[c][d]] - c_means[m]).norm(p=2).pow(2) / (2 * variance + 1e-8) - self.alpha ).exp() / denom[ind] + 1e-8).log() ).clamp(min=0.0) # adding epsilon 1e-8 inside log to prevent becoming inf at zero value

                loss_vector[c] -= (  ( -1 * (outputs[batch_clusters[c][d]] - c_means[m]).norm(p=2).pow(2) / (2 * variance + 1e-8) - self.alpha ).exp() / denom[ind] + 1e-8).log().clamp(min=0.0).cpu().data.numpy()[0]
                loss_count[c] += 1.0

        loss /= num_instances
        loss_vector /= num_instances

        return loss, loss_vector, loss_count, variance.cpu().data.numpy()[0]
