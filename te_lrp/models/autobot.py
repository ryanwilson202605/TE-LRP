import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR



class MapEncoderPts(nn.Module):
    """Encode map polylines into segment features conditioned on agent context."""

    def __init__(self, d_k, map_attr=3, dropout=0.1):
        """Build point projection, attention, and feed-forward layers.

        Inputs: d_k (int): hidden width; map_attr (int): map feature count; dropout (float).
        Returns: None."""
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.add = torch.add
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: init(
            m,
            nn.init.xavier_normal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2),
        )

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        """Build segment and point padding masks.

        Inputs: roads (Tensor[B,S,P,map_attr+1]): features with a final validity channel.
        Returns: tuple[Tensor, Tensor]: Boolean segment [B,S] and point [B*S,P] masks."""
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        # Retain the original model fallback for fully padded point rows.
        road_pts_mask = road_pts_mask.masked_fill((road_pts_mask.sum(-1) == roads.shape[2]).unsqueeze(-1), False)
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        """Encode map segments using the final agent-history embedding.

        Inputs: roads (Tensor[B,S,P,map_attr+1]), agents_emb (Tensor[T_obs,B,d_k]).
        Returns: tuple[Tensor, Tensor]: segment features [S,B,d_k] and padding mask [B,S]."""
        batch_size = roads.shape[0]
        num_segments = roads.shape[1]
        num_points = roads.shape[2]

        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(
            batch_size * num_segments,
            num_points,
            -1,
        ).permute(1, 0, 2)

        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, num_segments, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(
            query=agents_emb,
            key=road_pts_feats,
            value=road_pts_feats,
            key_padding_mask=road_pts_mask,
        )[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = self.add(input=road_seg_emb, other=self.map_feats(road_seg_emb))
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(batch_size, num_segments, -1)
        return road_seg_emb.permute(1, 0, 2), road_segment_mask


def init(module, weight_init, bias_init, gain=1):
    """Initialize a layer using explicit weight and bias initializers.

    Inputs: module (nn.Module), weight_init and bias_init (Callable), gain (float).
    Returns: nn.Module: the initialized input module."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    """Add sinusoidal temporal positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """Precompute a fixed sinusoidal position table.

        Inputs: d_model (int): feature width; dropout (float); max_len (int): maximum length.
        Returns: None."""
        super(PositionalEncoding, self).__init__()
        self.add = torch.add
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        """Add positional encoding and apply dropout to x (Tensor[T,B,d_model]); return a Tensor of the same shape."""
        x = self.add(input=x, other=self.pe[: x.size(0), :])
        return self.dropout(x)


class OutputModel(nn.Module):
    """Predict five bivariate trajectory-distribution parameters per time step."""

    def __init__(self, d_k=64):
        """Build the distribution-parameter MLP.

        Inputs: d_k (int): hidden feature width.
        Returns: None."""
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(
            m,
            nn.init.xavier_normal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2),
        )
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)),
            nn.ReLU(),
            init_(nn.Linear(d_k, d_k)),
            nn.ReLU(),
            init_(nn.Linear(d_k, 5)),
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        """Predict [mu_x, mu_y, sigma_x, sigma_y, rho] at each time step.

        Inputs: agent_decoder_state (Tensor[T,B*K,d_k]).
        Returns: Tensor[T,B*K,5]: distribution parameters."""
        num_steps = agent_decoder_state.shape[0]
        batch_modes = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(num_steps, batch_modes, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        # Bound correlation to stabilize covariance matrices.
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


class AutoBotEgo(nn.Module):
    """Predict multimodal ego trajectories using temporal, social, and map attention."""

    def __init__(self, config, k_attr=2, map_attr=2):
        """Build the AutoBot encoder, decoder, output heads, and loss.

        Inputs: config (DictConfig); k_attr and map_attr (int): agent and map feature counts.
        Returns: None."""
        super().__init__()
        self.config = config
        self.add = torch.add
        init_ = lambda m: init(
            m,
            nn.init.xavier_normal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2),
        )

        self.T = config.future_len
        self.past = config.past_len
        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = config.hidden_size
        self.c = config.num_modes

        self.L_enc = config.num_encoder_layers
        self.dropout = config.dropout
        self.num_heads = config.tx_num_heads
        self.L_dec = config.num_decoder_layers
        self.tx_hidden_size = config.tx_hidden_size

        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_k,
                nhead=self.num_heads,
                dropout=self.dropout,
                dim_feedforward=self.tx_hidden_size,
            )
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))
        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        self.map_encoder = MapEncoderPts(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)

        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.tx_decoder = []
        for _ in range(self.L_dec):
            self.tx_decoder.append(
                nn.TransformerDecoderLayer(
                    d_model=self.d_k,
                    nhead=self.num_heads,
                    dropout=self.dropout,
                    dim_feedforward=self.tx_hidden_size,
                )
            )
        self.tx_decoder = nn.ModuleList(self.tx_decoder)

        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past)
        self.output_model = OutputModel(d_k=self.d_k)

        self.P = nn.Parameter(torch.Tensor(self.c, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.P)

        self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)
        self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.criterion = Criterion(self.config)

    def generate_decoder_mask(self, seq_len, device):
        """Return a causal Boolean mask [seq_len,seq_len] for seq_len (int) and device (torch.device)."""
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        """Separate dynamic inputs from attention padding masks.

        Inputs: ego (Tensor[B,T_obs,k_attr+1]), agents (Tensor[B,T_obs,N,k_attr+1]).
        Returns: tuple[Tensor, ...]: ego and agent features, social masks, and decoder memory masks."""
        ego_tensor = ego[:, :, : self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)

        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)
        opps_tensor = agents[:, :, :, : self.k_attr]
        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        """Apply temporal self-attention independently to each agent.

        Inputs: agents_emb (Tensor[T_obs,B,N,d_k]), agent_masks (Tensor[B,T_obs,N]), layer (nn.Module).
        Returns: Tensor[T_obs,B,N,d_k]: temporal embeddings."""
        num_steps = agents_emb.size(0)
        batch_size = agent_masks.size(0)
        num_agents = agent_masks.size(2)
        temp_masks = agent_masks.permute(0, 2, 1).reshape(-1, num_steps)
        temp_masks = temp_masks.masked_fill((temp_masks.sum(-1) == num_steps).unsqueeze(-1), False)
        agents_temp_emb = layer(
            self.pos_encoder(agents_emb.reshape(num_steps, batch_size * num_agents, -1)),
            src_key_padding_mask=temp_masks,
        )
        return agents_temp_emb.view(num_steps, batch_size, num_agents, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        """Apply social self-attention independently at each time step.

        Inputs: agents_emb (Tensor[T_obs,B,N,d_k]), agent_masks (Tensor[B,T_obs,N]), layer (nn.Module).
        Returns: Tensor[T_obs,B,N,d_k]: social embeddings."""
        num_steps, batch_size, num_agents, _ = agents_emb.shape
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(num_agents, batch_size * num_steps, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, num_agents))
        agents_soc_emb = agents_soc_emb.view(num_agents, batch_size, num_steps, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def _forward(self, inputs):
        """Run the encoder and decoder to predict trajectories and mode probabilities.

        Inputs: inputs (dict[str, Tensor]): ego_in, agents_in, and roads with validity channels.
        Returns: dict[str, Tensor]: predicted_probability [B,K] and predicted_trajectory [B,K,T,5]."""
        ego_in, agents_in, roads = inputs["ego_in"], inputs["agents_in"], inputs["roads"]
        batch_size = ego_in.size(0)

        ego_tensor, agents_tensor_other, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), agents_tensor_other), dim=2)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)

        for i in range(self.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i])
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])
        ego_soctemp_emb = agents_emb[:, :, 0]

        orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)
        map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self.c, 1).view(-1, batch_size * self.c, self.d_k)
        road_segs_masks = orig_road_segs_masks.unsqueeze(1).repeat(1, self.c, 1).view(batch_size * self.c, -1)

        context = ego_soctemp_emb.unsqueeze(2).repeat(1, 1, self.c, 1)
        context = context.view(-1, batch_size * self.c, self.d_k)

        out_seq = self.Q.repeat(1, batch_size, 1, 1).view(self.T, batch_size * self.c, -1)
        time_masks = self.generate_decoder_mask(seq_len=self.T, device=ego_in.device)
        for d in range(self.L_dec):
            self.map_attn_layers.te_lrp_iiq = d == 0
            ego_dec_emb_map = self.map_attn_layers(
                query=out_seq,
                key=map_features,
                value=map_features,
                key_padding_mask=road_segs_masks,
            )[0]
            out_seq = self.add(input=out_seq, other=ego_dec_emb_map)
            out_seq = self.tx_decoder[d](out_seq, context, tgt_mask=time_masks, memory_key_padding_mask=env_masks)
        out_dists = self.output_model(out_seq).reshape(self.T, batch_size, self.c, -1).permute(2, 0, 1, 3)

        mode_params_emb = self.P.repeat(1, batch_size, 1)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=ego_soctemp_emb, value=ego_soctemp_emb)[0]
        mode_params_emb = self.mode_map_attn(
            query=mode_params_emb,
            key=orig_map_features,
            value=orig_map_features,
            key_padding_mask=orig_road_segs_masks,
        )[0] + mode_params_emb
        mode_probs = F.softmax(self.prob_predictor(mode_params_emb).squeeze(-1), dim=0).transpose(0, 1)

        output = {}
        output["predicted_probability"] = mode_probs
        output["predicted_trajectory"] = out_dists.permute(2, 0, 1, 3)
        return output

    def forward(self, batch):
        """Run training forward on batch (dict); return predictions (dict) and loss (Tensor)."""
        output = self.predict(batch=batch)
        return output, self.get_loss(batch=batch, prediction=output)

    def predict(self, batch):
        """Predict using history and map inputs without reading ground truth.

        Inputs: batch (dict): trajectories, maps, masks, and target indices.
        Returns: dict[str, Tensor]: trajectories and mode probabilities."""
        model_input = {}
        inputs = batch["input_dict"]

        agents_in = inputs["obj_trajs"]
        agents_mask = inputs["obj_trajs_mask"]
        roads = inputs["map_polylines"]

        ego_in = torch.gather(
            agents_in,
            1,
            inputs["track_index_to_predict"].view(-1, 1, 1, 1).repeat(1, 1, *agents_in.shape[-2:]),
        ).squeeze(1)
        ego_mask = torch.gather(
            agents_mask,
            1,
            inputs["track_index_to_predict"].view(-1, 1, 1).repeat(1, 1, agents_mask.shape[-1]),
        ).squeeze(1)

        # AutoBot consumes xy coordinates and validity channels.
        agents_in = torch.cat([agents_in[..., :2], agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in[..., :2], ego_mask.unsqueeze(-1)], dim=-1)

        roads = torch.cat([roads[..., :2], inputs["map_polylines_mask"].unsqueeze(-1)], dim=-1)

        model_input["ego_in"] = ego_in
        model_input["agents_in"] = agents_in
        model_input["roads"] = roads

        output = self._forward(model_input)
        return output

    def get_loss(self, batch, prediction):
        """Compute the criterion from predictions and ground-truth fields.

        Inputs: batch (dict): targets and masks; prediction (dict[str, Tensor]).
        Returns: Tensor: scalar loss."""
        inputs = batch["input_dict"]
        ground_truth = torch.cat([inputs["center_gt_trajs"][..., :2], inputs["center_gt_trajs_mask"].unsqueeze(-1)], dim=-1)
        loss = self.criterion(prediction, ground_truth, inputs["center_gt_final_valid_idx"])
        return loss

    def configure_optimizers(self):
        """Build the optimizer and learning-rate scheduler; return two lists in a tuple."""
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate, eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config.learning_rate_sched, gamma=0.5)
        return [optimizer], [scheduler]


class Criterion(nn.Module):
    """Combine multimodal NLL, entropy, KL, and optional ADE/FDE losses."""

    def __init__(self, config):
        """Store loss weights and options from config (DictConfig); return None."""
        super(Criterion, self).__init__()
        self.config = config

    def forward(self, out, gt, center_gt_final_valid_idx):
        """Compute the full multimodal training loss.

        Inputs: out (dict[str, Tensor]), gt (Tensor), center_gt_final_valid_idx (Tensor).
        Returns: Tensor: scalar loss."""
        return self.nll_loss_multimodes(out, gt, center_gt_final_valid_idx)

    def get_BVG_distributions(self, pred):
        """Build bivariate Gaussian distributions for entropy regularization.

        Inputs: pred (Tensor[B,T,5]): means, scales, and correlation.
        Returns: MultivariateNormal: batched distributions."""
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        cov = torch.stack(
            [
                torch.stack([sigma_x ** 2, rho * sigma_x * sigma_y], dim=-1),
                torch.stack([rho * sigma_x * sigma_y, sigma_y ** 2], dim=-1),
            ],
            dim=-2,
        )
        biv_gauss_dist = MultivariateNormal(
            loc=torch.cat((mu_x, mu_y), dim=-1),
            covariance_matrix=cov,
            validate_args=False,
        )
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        """Return x/y Laplace distributions from pred (Tensor[B,T,5])."""
        return Laplace(pred[:, :, :2], pred[:, :, 2:4], validate_args=False)

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        """Compute masked trajectory NLL or log likelihood.

        Inputs: pred (Tensor[B,T,5]), data (Tensor[B,T,3]), mask (Tensor[B,T]), rtn_loss (bool).
        Returns: Tensor[B]: NLL if rtn_loss is true, otherwise log likelihood."""
        biv_gauss_dist = self.get_Laplace_dist(pred)
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)
        return ((biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx):
        """Combine posterior-weighted NLL with entropy, KL, and auxiliary losses.

        Inputs: output (dict[str, Tensor]), data (Tensor[B,T,3]), center_gt_final_valid_idx (Tensor).
        Returns: Tensor: scalar training objective."""
        modes_pred = output["predicted_probability"]
        pred = output["predicted_trajectory"].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config.entropy_weight
        kl_weight = self.config.kl_weight
        use_FDEADE_aux_loss = self.config.use_FDEADE_aux_loss

        modes = len(pred)

        log_lik_list = []
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik_list.append(nll.unsqueeze(1))

            log_lik = torch.cat(log_lik_list, dim=1)
            log_priors = torch.log(modes_pred)
            log_posterior_unnorm = log_lik + log_priors
            logsumexp = torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
            log_posterior = log_posterior_unnorm - logsumexp
            post_pr = torch.exp(log_posterior).to(data.device)

        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Penalize high within-mode uncertainty.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # Match prior mode probabilities to posterior responsibilities.
        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        final_loss = loss + kl_loss + adefde_loss
        return final_loss

    def l2_loss_fde(self, pred, data, mask):
        """Compute the best-mode ADE/FDE auxiliary loss.

        Inputs: pred (Tensor[K,T,B,5]), data (Tensor[B,T,3]), mask (Tensor[B,T]).
        Returns: Tensor: scalar auxiliary loss scaled by 100."""
        fde_loss = torch.norm(
            pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1),
            2,
            dim=-1,
        ) * mask[:, -1:]
        ade_loss = torch.norm(
            pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0),
            2,
            dim=-1,
        )
        ade_loss = (ade_loss * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, _ = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()
