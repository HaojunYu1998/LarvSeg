# import torch
# from torch.nn import functional as F
# import numpy as np

# try:
#     from pygco import cut_from_graph
# except ImportError:
#     raise FileNotFoundError(
#         "Missing Grah-Cut (GCO) library,"
#         " please install it from https://github.com/Borda/pyGCO."
#     )


# def unary_from_logits(
#     logits, 
#     pix_embedding, 
#     unary_mul,
#     background_cost,
#     temperature, 
#     unary_from_embedding
# ):
#     H, W, C = pix_embedding.shape
#     N_cls = logits.shape[0]
#     pix_embedding = pix_embedding.reshape(H * W, C)
#     # TODO: logits thresholds
#     # convert logits to probabilities
#     probs = F.softmax(logits / temperature, dim=0)
#     # (N_cls, H * W)
#     probs = probs.reshape(N_cls, H * W)
#     num_priors = (probs == probs.max(dim=0).values[None, :]).sum(dim=1).float()
#     prior_mask = probs.max(dim=0).indices
#     # (N_prior, )
#     prior_cls = num_priors.nonzero(as_tuple=False).flatten()
#     if unary_from_embedding:
#         probs = probs[prior_cls]
#         probs = probs.permute(0, 1)
#     else:
#         # (N_prior, )
#         cos_sims = []
#         background = torch.zeros(H * W).to(probs.device) + background_cost
#         if len(prior_cls) > 0:
#             pix_embedding = pix_embedding.reshape(H * W, C)
#             for cls in prior_cls:
#                 cos_sim = pix_embedding[prior_mask == cls] @ pix_embedding.T
#                 cos_sim = cos_sim.max(dim=0).values
#                 cos_sims.append(cos_sim)
#         else:
#             del probs, num_priors
#         cos_sims.append(background)
#     # (H * W, N_prior)
#     probs = torch.stack(cos_sims, dim=-1)
#     unary_cost = probs.max(dim=-1).values[:, None] - probs
#     unary_cost *= unary_mul
#     return unary_cost.cpu(), prior_cls.cpu()


# def edge_weight_from_embedding(pix_embedding, edges, eta=2, gamma=1.0, eps=1e-8):
#     h, w, c = pix_embedding.shape
#     pix_embedding = pix_embedding.reshape(h * w, c)
#     dist_norm = torch.norm(
#         pix_embedding[torch.from_numpy(edges[:, 0]).to(pix_embedding.device)]
#         - pix_embedding[torch.from_numpy(edges[:, 1]).to(pix_embedding.device)],
#         p=2, dim=-1
#     )
#     edge_weight = 1.0 / (eps + torch.pow(dist_norm, 2 * eta))
#     edge_weight = edge_weight.detach().cpu().numpy()
#     return edge_weight.reshape(-1, 1) * gamma


# def construct_neightbor_edges(edge_type, shape):
#     """
#     point_x: [N_Points,1]
#     point_y: [N_Points,1]
#     x_shift: [N_Neighbors,]
#     y_shift: [N_Neighbors,]
#     """
#     h, w = shape
#     point_x = np.arange(w)
#     point_y = np.arange(h)
#     point_x, point_y = np.meshgrid(point_x, point_y)
#     point_x = point_x.reshape((-1, 1))
#     point_y = point_y.reshape((-1, 1))

#     if edge_type == "four_connect":
#         x_shift = np.array([[1, 0]])
#         y_shift = np.array([[0, 1]])
#     elif edge_type == "eight_connect":
#         x_shift = np.array([[-1, 1, -1, 0, 1]])
#         y_shift = np.array([[0, 0, 1, 1, 1]])

#     x_shift = x_shift.reshape((1, -1))
#     y_shift = y_shift.reshape((1, -1))
#     point_x = point_x.reshape((-1, 1))
#     point_y = point_y.reshape((-1, 1))

#     end_x = point_x + x_shift  # N_Points,N_Neighbors
#     end_y = point_y + y_shift  # N_Points,N_Neighbors
#     start_x = np.repeat(point_x, x_shift.shape[-1], axis=1)  # N_Points,N_Neighbors
#     start_y = np.repeat(point_y, y_shift.shape[-1], axis=1)  # N_Points,N_Neighbors

#     start = start_y * w + start_x  # N_Points,N_Neighbors
#     end = end_y * w + end_x  # N_Points,N_Neighbors
#     start = start.reshape((-1, 1))
#     end = end.reshape((-1, 1))
#     edges = np.concatenate((start, end), axis=1)  # N_Points*N_Neighbors,2

#     num_nodes = h * w
#     valid_start = (edges[:, 0] >= 0) & (edges[:, 0] < num_nodes)
#     valid_end = (edges[:, 1] >= 0) & (edges[:, 1] < num_nodes)
#     valid_mask = valid_start & valid_end
#     edges = edges[valid_mask]
#     return edges


# def graph_cut_post_process(
#     logits,
#     pix_embedding,
#     unary_mul=1000,
#     logits_temperature=1,
#     background_cost=0.0,
#     unary_only=False,
#     unary_from_embedding=False,
#     pairwise_weight=False,
#     n_iter=5,
#     eps=1e-8,
#     edge_type="four_connect",
#     label_compatibility=None,
#     label_transition_cost=1,
# ):
#     """
#     logits: [N_Cls,H,W], logits for each pixel
#     pix_embedding: [H,W,C], embedding for each pixel
#     label_compatibility: [N_Cls,N_Cls], similarity between each label
#     n_labels: int
#     """
#     n_labels, h, w = logits.shape
#     # defina the unary energy of each pixel
#     (
#         unary_cost, # [H * W, N_prior]
#         prior_cls, # [N_prior, ]
#     ) = unary_from_logits(
#         logits=logits,
#         pix_embedding=pix_embedding,
#         unary_mul=unary_mul,
#         background_cost=background_cost,
#         temperature=logits_temperature, 
#         unary_from_embedding=unary_from_embedding,
#     )
    
#     if unary_only:
#         graph_labels = unary_cost.argmin(dim=-1)
#     else:
#         n_priors = unary_cost.shape[-1]
#         if label_compatibility is None:
#             pairwise_cost = (np.ones(n_priors) - np.eye(n_priors)) * label_transition_cost
#         else:
#             if isinstance(label_compatibility, torch.Tensor):
#                 label_compatibility = label_compatibility.detach().cpu().numpy()
#             pairwise_cost = (
#                 1 / (label_compatibility + eps) - np.eye(n_priors)
#             ) * label_transition_cost

#         # define the graph with edges
#         edges = construct_neightbor_edges(edge_type, (h, w))
#         if pairwise_weight:
#             gamma = 1e-3 / (np.sqrt(h * w))
#             edge_weight = edge_weight_from_embedding(
#                 pix_embedding, edges, gamma=gamma
#             )
#             edges = np.concatenate([edges, edge_weight], axis=1)

#         graph_labels = cut_from_graph(
#             np.ascontiguousarray(edges).astype(np.int32),
#             np.ascontiguousarray(unary_cost).astype(np.int32),
#             np.ascontiguousarray(pairwise_cost).astype(np.int32),
#             algorithm="swap",
#             n_iter=n_iter,
#         )
#         graph_labels = torch.from_numpy(graph_labels).long()

#     graph_labels = graph_labels.reshape(h, w)
#     graph_labels_ = torch.zeros_like(graph_labels)
#     for i, cls in enumerate(prior_cls):
#         graph_labels_[graph_labels == i] = cls

#     return graph_labels_