import torch
from typing import Tuple, Callable, Optional, Union
import torch.nn.functional as F

@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def token_merge_bipartite2d_multi_batch(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    info_map=None,  # info_map [N,1,Hp,Wp]
):

    B, N, C = metric.shape
    per_batch_ops = []

    if info_map is not None:
        img_per_batch = info_map.shape[0] // B 
        info_map = info_map.view(B, img_per_batch, *info_map.shape[1:]) 

    for b in range(B):
        metric_b = metric[b:b+1]  # shape [1, N, C]
        info_map_b = info_map[b:b+1].squeeze(0) if info_map is not None else None

        merge_b, unmerge_b = token_merge_bipartite2d(
            metric_b, w, h, sx, sy, r,
            no_rand=no_rand,
            generator=generator,
            enable_protection=enable_protection,
            info_map=info_map_b,
        )
        per_batch_ops.append((merge_b, unmerge_b))

    def merge(x: torch.Tensor, mode: str = "mean", extra_tensors=None, extra_tensors_2=None):
        results_main, results_extra1, results_extra2 = [], [], []
        for b in range(B):
            m_b, _ = per_batch_ops[b]
            out = m_b(
                x[b:b+1],
                mode=mode,
                extra_tensors=None if extra_tensors is None else extra_tensors[b:b+1],
                extra_tensors_2=None if extra_tensors_2 is None else extra_tensors_2[b:b+1],
            )
            if isinstance(out, tuple):
                results_main.append(out[0])
                if len(out) > 1:
                    results_extra1.append(out[1])
                if len(out) > 2:
                    results_extra2.append(out[2])
            else:
                results_main.append(out)

        main = torch.cat(results_main, dim=0)
        if results_extra1 and results_extra2:
            return main, torch.cat(results_extra1, dim=0), torch.cat(results_extra2, dim=0)
        elif results_extra1:
            return main, torch.cat(results_extra1, dim=0)
        else:
            return main

    def unmerge(x: torch.Tensor):
        results = []
        for b in range(B):
            _, u_b = per_batch_ops[b]
            results.append(u_b(x[b:b+1]))
        return torch.cat(results, dim=0)

    return merge, unmerge


def compute_info_maps(
    images_normed: torch.Tensor,   # [N, 3, H, W]  
    patch_tokens: torch.Tensor,    # [N, P, C] 
    var_win: int = 3,
    proj_dim: int = 32,
):

    images_normed = images_normed.to(torch.float32)
    patch_tokens = patch_tokens.to(torch.float32)
    device = patch_tokens.device
    N, P, C = patch_tokens.shape
    H = images_normed.shape[-2]
    W = images_normed.shape[-1]
    patch_size = 14
    Hp = H // patch_size
    Wp = W // patch_size
    Hc, Wc = Hp * patch_size, Wp * patch_size
    assert P == Hp * Wp, f"P={P} vs Hp*Wp={Hp*Wp}"

    tok = patch_tokens.view(N, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()  # [N,C,Hp,Wp]

    with torch.random.fork_rng(devices=[device]):
        Pmat = torch.empty(C, proj_dim, device=device)
        torch.nn.init.orthogonal_(Pmat)
    X = torch.einsum('nchw,cd->ndhw', tok, Pmat)   # [N,d,Hp,Wp]

    pad = var_win // 2 
    mu = F.avg_pool2d(X, kernel_size=var_win, stride=1, padding=pad, count_include_pad=False)       # [N,d,Hp,Wp]
    m2 = F.avg_pool2d(X*X, kernel_size=var_win, stride=1, padding=pad, count_include_pad=False)     # [N,d,Hp,Wp]
    var_map = (m2 - mu*mu).clamp_min(0.0).sum(dim=1, keepdim=True)         # [N,1,Hp,Wp]

    x = images_normed[:, :, :Hc, :Wc]           # [N,3,Hc,Wc]
    gray = (0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3])
    # Sobel
    kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device=device).view(1,1,3,3)
    ky = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device=device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(gx*gx + gy*gy + 1e-12)    # [N,1,Hc,Wc]
    grad_map_tok = F.adaptive_avg_pool2d(grad, (Hp, Wp))  # [N,1,Hp,Wp]

    def norm01(t):
        tmin = t.amin(dim=(-2,-1), keepdim=True)
        tmax = t.amax(dim=(-2,-1), keepdim=True)
        return (t - tmin) / (tmax - tmin + 1e-8)

    var_n  = norm01(var_map)
    grad_n = norm01(grad_map_tok)
    info   = 0.3* var_n + 0.7* grad_n                 # [N,1,Hp,Wp]
    info_n = norm01(info)
    gamma = 1.4  
    info_n = info_n ** gamma
    info_up = F.interpolate(info_n, size=(Hc, Wc), mode='bilinear', align_corners=False)  # [N,1,Hc,Wc]

    return {
        "var_map": var_map,               # [N,1,Hp,Wp]
        "grad_map_tok": grad_map_tok,     # [N,1,Hp,Wp]
        "info_map": info_n.to(torch.bfloat16),               # [N,1,Hp,Wp]
        "info_up": info_up,               # [N,1,Hc,Wc]
        "Hp": Hp, "Wp": Wp, "Hc": Hc, "Wc": Wc
    }

def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    info_map=None,
) -> Tuple[Callable, Callable]:
    """
    Divide tokens into source (src) and destination (dst) groups, and merge r tokens from src to dst.
    dst tokens are selected by randomly choosing one token from each (sx, sy) region.
    Optionally protect the top 10% of tokens from merging based on importance scores.

    Args:
     - metric [B, N, C]: Tensor for similarity computation, B=batch size, N=token count, C=feature dimension
     - w: Image width in tokens
     - h: Image height in tokens
     - sx: dst stride in x dimension, must divide w evenly
     - sy: dst stride in y dimension, must divide h evenly
     - r: Number of tokens to remove through merging
     - no_rand: If True, disable randomness (use only top-left token)
     - generator: Random number generator if no_rand is False and not None
     - enable_protection: If True, enable importance protection feature

    Returns:
     - (merge, unmerge): Two functions for merging tokens and restoring pre-merge state
    """

    ## B，S*N ,C
    B, N, _ = metric.shape  # Batch size B, total tokens N
    if r <= 0:
        return do_nothing, do_nothing
    

    gather = torch.gather

    tokens_per_img = w * h + 5
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, "Token count doesn't match (w*h+5)*num_imgs"

    with torch.no_grad():
        # Determine whether to compute importance scores based on enable_protection
        if enable_protection:

            ## info_map [N,1,Hp,Wp]
            if info_map is not None:
                protect_ratio = 0.1
                info = info_map[:, 0].to(metric.device)  # [num_imgs, Hp, Wp]
                k = max(1, int(info.shape[-2] * info.shape[-1] * protect_ratio))
                topk_idx = info.flatten(1).topk(k, dim=1).indices  # [num_imgs, k]
                offsets = torch.arange(num_imgs, device=info.device) * tokens_per_img + 5

                protected_indices = (topk_idx + offsets[:, None]).flatten()
                num_protected = protected_indices.numel()

            else:
                num_protected = int(N * 0.1)
                step = max(1, N // num_protected)
                protected_indices = torch.arange(0, N, step, device=metric.device)[
                    :num_protected
                ]

        else:
            protected_indices = None
            num_protected = 0


        # Global idx_buffer_seq of length N; -1 indicates dst, 0 indicates src (maintain original logic)
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx  # Number of blocks within each image

        # Mark first image entirely as dst
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        # Process other images - fully vectorized batch operations
        if num_imgs > 1:
            
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
            idx_buffer_seq[cls_indices.flatten()] = -1

            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img + 5
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
            else:
                total_other_imgs = num_imgs - 1

                if info_map is not None:
                    info_map_other_imgs = info_map[1:, 0]  # [num_imgs-1, Hp, Wp]
                    Hp, Wp = info_map_other_imgs.shape[-2:]
                    valid_h = (Hp // sy) * sy
                    valid_w = (Wp // sx) * sx
                    info_valid = info_map_other_imgs[:, :valid_h, :valid_w]

                    all_rand_idx = (
                        info_valid
                        .view(total_other_imgs, valid_h // sy, sy, valid_w // sx, sx)
                        .reshape(total_other_imgs, valid_h // sy, valid_w // sx, sy * sx)
                        .argmin(dim=-1)
                    )
                else:
                    all_rand_idx = torch.randint(
                        sy * sx,
                        size=(total_other_imgs, hsy, wsx),
                        device=metric.device,
                        generator=generator,
                    )

                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )

                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )

                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )

                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                # Batch fill to target positions - still needs a small loop here, but operations are greatly reduced
                for i in range(total_other_imgs):
                    img_idx = i + 1
                    grid_start = img_idx * tokens_per_img + 5
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        flat_view
                    )

        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        # Original src and dst indices
        a_idx_orig = rand_idx[:, num_dst_orig:, :]
        b_idx_orig = rand_idx[:, :num_dst_orig, :]
        a_idx = a_idx_orig
        b_idx = b_idx_orig

        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        # Define an internal function to separate src, dst, and protected tokens
        def split(x):
            C = x.shape[-1]

            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

        # Compute cosine similarity (normalize first then dot product)
        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, protected = split(metric)
        else:
            a, b = split(metric)

        r = min(a.shape[1], r)

        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

        b_transposed = b.transpose(-1, -2)
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # If protection is enabled, filter out protected tokens to ensure they are not merged
        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            valid_mask = ~protected_mask_src[edge_flat]
            valid_edges = edge_flat[valid_mask]

            valid_count = valid_edges.shape[0]
            r_actual = min(r, valid_count)

            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            r_actual = r

        # Get dst token indices corresponding to each src token to be merged
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        r = r_actual


    # Define merge function to merge selected src tokens to corresponding dst tokens
    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, t1, c = src.shape

        # Extract unmerged src tokens - using actual unm_idx size
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        # ---------------- Extra tensor processing ----------------
        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            E_dim = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)

            # Consistent with main tensor, only select r src tokens to be merged
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, E_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, E_dim))

            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode
            )
            if enable_protection:
                merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
            else:
                merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)

        if extra_tensors_2 is not None:
            E_dim_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)

            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, E_dim_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, E_dim_2))

            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode
            )
            if enable_protection:
                merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
            else:
                merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)

        if enable_protection:
            main_result = torch.cat([unm, dst, protected], dim=1)
        else:
            main_result = torch.cat([unm, dst], dim=1)

        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        else:
            return main_result

    # Define unmerge function to restore pre-merge state (for decoder)
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]

        if enable_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )

        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )

        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    return merge, unmerge

