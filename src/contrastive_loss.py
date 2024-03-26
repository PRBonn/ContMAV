########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################

import torch
import torch.nn.functional as F


def ContrastiveLoss(emb_k, emb_q, labels, tau=0.1):
    """
    emb_k: the feature bank with the aggregated embeddings over the iterations
    emb_q: the embeddings for the current iteration
    labels: the correspondent class labels for each sample in emb_q
    """
    import ipdb;ipdb.set_trace()  # fmt: skip
    assert emb_q.shape[0] == labels.shape[0], "mismatch on emb_q and labels shapes!"
    emb_k1 = F.normalize(emb_k, dim=-1)
    emb_q1 = F.normalize(emb_q, dim=-1)

    emb_k2 = torch.sigmoid(emb_k)
    emb_q2 = torch.sigmoid(emb_q)
    # temperature-scaled cosine similarity
    sim_qk1 = (emb_q1 @ emb_k1.T) / tau
    sim_qk2 = (emb_q2 @ emb_k2.T) / tau

    print(F.cross_entropy(sim_qk1, labels))
    print(F.cross_entropy(sim_qk2, labels))
    return F.cross_entropy(sim_qk1, labels)


emb_q = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
emb_k = torch.tensor(
    [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0], [3.0, 1.0, 2.0]]
)
labels = torch.tensor([0.0, 1.0])

loss = ContrastiveLoss(emb_k, emb_q, labels.long())
print(loss)
