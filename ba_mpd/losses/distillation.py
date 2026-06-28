import torch
import torch.nn.functional as F


def boundary_anchor(teacher_logits, labels, top_k, temperature=2.0):
    """Swap the true-label mass with the top-k boundary mass when needed.

    The operation is deterministic and label-aware but not label-dominant:
    if the ground-truth label is absent from teacher top-k, it receives only
    the boundary class probability, while the teacher top-(k-1) ranking is
    unchanged.
    """

    tau = float(temperature)
    teacher_p = F.softmax(teacher_logits / tau, dim=1)
    top_k = min(max(int(top_k), 1), teacher_logits.size(1) - 1)
    top_indices = teacher_p.topk(top_k, dim=1).indices
    in_top = top_indices.eq(labels.long().view(-1, 1)).any(dim=1)
    boundary = top_indices[:, -1]

    anchored_p = teacher_p.clone()
    anchored_indices = top_indices.clone()
    rows = torch.arange(teacher_logits.size(0), device=teacher_logits.device)
    replace_rows = rows[~in_top]
    if replace_rows.numel() > 0:
        y = labels.long()[replace_rows]
        c_k = boundary[replace_rows]
        anchored_indices[replace_rows, -1] = y
        anchored_p[replace_rows, y] = teacher_p[replace_rows, c_k]
        anchored_p[replace_rows, c_k] = teacher_p[replace_rows, y]

    support = torch.zeros_like(teacher_p, dtype=torch.bool)
    support.scatter_(1, anchored_indices, True)
    return anchored_p.detach(), support


def _masked_relation_kl(student_p, teacher_p, mask):
    student_region = student_p.masked_fill(~mask, 0.0)
    teacher_region = teacher_p.masked_fill(~mask, 0.0)
    student_region = student_region / student_region.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_region = teacher_region / teacher_region.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(student_region.clamp_min(1e-8).log(), teacher_region.detach(), reduction="batchmean")


def mpd_loss(student_logits, teacher_logits, top_k=5, temperature=2.0, alpha=2.0, teacher_probs=None, support=None):
    """Mass-Partitioned Distillation without boundary anchoring.

    The objective is tau^2 * (L_mass + L_top + alpha * L_comp).
    """

    tau = float(temperature)
    student_p = F.softmax(student_logits / tau, dim=1)
    if teacher_probs is None:
        teacher_p = F.softmax(teacher_logits / tau, dim=1)
    else:
        teacher_p = teacher_probs

    if support is None:
        top_k = min(max(int(top_k), 1), student_logits.size(1) - 1)
        top_indices = teacher_p.topk(top_k, dim=1).indices
        support = torch.zeros_like(teacher_p, dtype=torch.bool)
        support.scatter_(1, top_indices, True)

    complement = ~support
    student_mass = student_p.masked_fill(~support, 0.0).sum(dim=1).clamp(1e-8, 1.0 - 1e-8)
    teacher_mass = teacher_p.masked_fill(~support, 0.0).sum(dim=1).clamp(1e-8, 1.0 - 1e-8)
    student_rho = torch.stack([student_mass, 1.0 - student_mass], dim=1)
    teacher_rho = torch.stack([teacher_mass, 1.0 - teacher_mass], dim=1)

    loss_mass = F.kl_div(student_rho.log(), teacher_rho.detach(), reduction="batchmean")
    loss_top = _masked_relation_kl(student_p, teacher_p, support)
    loss_comp = _masked_relation_kl(student_p, teacher_p, complement)
    return (loss_mass + loss_top + float(alpha) * loss_comp) * (tau * tau)


def ba_mpd_loss(student_logits, teacher_logits, labels, top_k=5, temperature=2.0, alpha=2.0):
    """Boundary-Anchored Mass-Partitioned Distillation."""

    teacher_p, support = boundary_anchor(teacher_logits, labels, top_k=top_k, temperature=temperature)
    return mpd_loss(
        student_logits,
        teacher_logits,
        top_k=top_k,
        temperature=temperature,
        alpha=alpha,
        teacher_probs=teacher_p,
        support=support,
    )
