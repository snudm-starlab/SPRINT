"""
File: CompresseionManager.py
- Implementation of CompressionManager that supports compression-related functions
- This class supports importance scoring, latency estimation, and the selection of 
  pruning targets
"""

import torch
from utils.general_utils import *

class CompressionManager(object):
    """
    A class for managing compression status of models.
    This class supports importance scoring, latency estimation, and the selection of 
    pruning targets 
    """
    def __init__(self, num_sublayers, nsamples, metrics, latency_dict, latency_aware):
        self.num_sublayers = num_sublayers
        self.nsamples = nsamples
        self.metrics = metrics
        self.imp_scores = {k:torch.zeros(num_sublayers) for k in metrics}
        self.imp_score_samples = {k:0 for k in metrics}
        self.pruning_status = [True for _ in range(num_sublayers)] # binary pruning status
        self.pruned_sublayers = []
        self.tuning_indices = list(range(num_sublayers))
        self.latency_dict = latency_dict
        self.dense_latency = self.get_latency([True for _ in range(self.num_sublayers)])
        self.latency_aware = latency_aware
        
    def reinit_imp_scores(self, subid=None):
        """
        Reinitialize importance scores

        Args:
            subid (int, optional): the index of sublayer to reinit. Defaults to None.
        """
        if subid is None:
            for _metric in self.imp_scores.keys():
                self.imp_scores[_metric] = 0.
        else:
            for _metric in self.imp_scores.keys():
                self.imp_scores[_metric][subid] = 0.
        for _metric in self.imp_scores.keys():
            self.imp_score_samples[_metric] = 0
    
    def add_batch(self, subid: int, origin: torch.Tensor, compressed: torch.Tensor):
        """
        Update importance scores using a batch of activations

        Args:
            subid (int): the sublayer index to update importance score
            origin (torch.Tensor): activations of the unpruned model
            compressed (torch.Tensor): activations of the pruned model
        """
        assert len(origin.shape) == 3, "Only support 3D activations"
        n, s, d = origin.shape
        # Updating importance scores for each metric
        for _metric in self.metrics:
            imp_score_type, norm_type = _metric.split('_')

            # Reshaping activations according to the normalization type
            if norm_type == 'sentence':
                _origin = origin.view(n, -1).float()
                _compressed = compressed.view(n, -1).float()
            elif norm_type == 'token':
                _origin = origin.view(n*s, -1).float()
                _compressed = compressed.view(n*s, -1).float()
            elif norm_type == 'channel':
                _origin = torch.permute(origin, (0,2,1))
                _origin = _origin.reshape(n*d, -1).float()
                _compressed = torch.permute(compressed, (0,2,1))
                _compressed = _compressed.reshape(n*d, -1).float()
            elif norm_type == 'element':
                # Return nan because of zeros
                _origin = origin.view(n*s*d, -1).float()
                _compressed = compressed.view(n*s*d, -1).float()
            elif norm_type is None or norm_type == "None":
                _origin = origin.view(1, -1).float()
                _compressed = compressed.view(1, -1).float()
            else:
                raise Exception("Unknown type of sensivity normalization." +
                                "You must select a normalization type in ['sentence', 'token', 'channel', 'element', 'None']")
            self.imp_score_samples[_metric] += _origin.shape[0]
        
            if "l1" in imp_score_type: 
                self.imp_scores[_metric][subid] += self.add_batch_l1(_origin, _compressed).cpu()
            if "l2" in imp_score_type: 
                self.imp_scores[_metric][subid] += self.add_batch_l2(_origin, _compressed).cpu()
            if "cos" in imp_score_type: 
                self.imp_scores[_metric][subid] += self.add_batch_cos(_origin, _compressed).cpu()
    
    def add_batch_l1(self, origin, compressed):
        """
        Update importance score using l1 norm
        """
        _imp_score = torch.abs(origin-compressed).sum(dim=1)
        _norm = torch.abs(origin).sum(dim=1)
        imp_score =( _imp_score / _norm).sum()
        return imp_score
    
    def add_batch_l2(self, origin, compressed):
        """
        Update importance score using l2 norm
        """
        _imp_score = (torch.sqrt((origin-compressed)**2)).sum(dim=1)
        _norm = (torch.sqrt((origin)**2)).sum(dim=1)
        imp_score =(_imp_score / _norm).sum()
        return imp_score
    
    def add_batch_cos(self, origin, compressed):
        """
        Update importance score using cos dissimilarity
        """
        orig_norm = torch.sqrt((origin**2).sum(dim=1))
        comp_norm = torch.sqrt((compressed**2).sum(dim=1))
        _imp_score = 1 - (origin * compressed).sum(dim=1) / (orig_norm * comp_norm)
        imp_score = _imp_score.sum()
        return imp_score
    
    def is_pruned(self, subid):
        """
        Return whether the sublayer is pruned or not
        """
        return not self.pruning_status[subid]
    
    def update_status(self, subid, status):
        """
        Update the pruning status of a sublayer
        """
        self.pruning_status[subid] = status
        if status == False:
            self.set_score_inf(subid)
            self.pruned_sublayers.append(subid)
    
    def set_score_inf(self, subid=None):
        """
        Setting an infinity importance score for a sublayer.
        This function is usually called for pruned sublayers.
        """
        for _metric in self.imp_scores.keys():
            if subid is None:
                for _i in range(len(self.imp_scores[_metric])):
                    self.imp_scores[_metric][_i] = float('inf')
            else:
                self.imp_scores[_metric][subid] = float('inf')
    
    def get_target_subids(self, num_prune:int):
        """
        Find the sublayers to prune.
        Args:
            num_prune (int): the number of sublayers to select. This argument is used for candidate selection.

        Returns:
            comp_target_subids (list): the list of selected sublayers
        """
        total_rank = None
        for _metric in self.imp_scores.keys():
            if self.latency_aware:
                _imp_scores = torch.Tensor(self.imp_scores[_metric]).view(-1, 2)
                _imp_scores = (_imp_scores * torch.tensor([1/self.latency_dict["MHA"], 
                    1/self.latency_dict["MLP"]])).view(-1)
                _rank = torch.sort(torch.sort(_imp_scores).indices).indices
            else:
                _rank = torch.sort(torch.sort(torch.Tensor(self.imp_scores[_metric])).indices).indices

            if total_rank is None:
                total_rank = _rank
            else:
                total_rank += _rank

        comp_target_subids = torch.sort(total_rank, stable=True).indices[:num_prune].tolist()
        return comp_target_subids
    
    
    def get_pruned_sublayer_list(self):
        """
        Return the indices of pruned sublayers
        """
        return self.pruned_sublayers

    def get_latency(self, _pruning_status):
        """
        Estimate the latency of a pruned model based on the given _pruning status

        Args:
            _pruning_status (dict): pruning status of sublayers (1: unpruned, 0: pruned).

        Returns:
            _lat (float): the estimated latency
        """
        _lat = 0
        for sid in range(self.num_sublayers):
            if _pruning_status[sid] is True:
                if sid % 2 ==0:
                    _lat += self.latency_dict["MHA"]
                else:
                    _lat += self.latency_dict["MLP"]
        return _lat

    def get_speedup(self, ):
        """
        Estimate the speedup of a pruned model
        """
        pruned_latency = self.get_latency(self.pruning_status)
        return self.dense_latency / pruned_latency
    