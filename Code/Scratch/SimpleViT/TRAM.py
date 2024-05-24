import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention_simple(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer_simple(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_simple(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def get_mask(self, matrices, centrality_prec, n_patch,idx):

        centrality_values_in = torch.sum(matrices, dim=1)  # in-degree (batch, number_nodes)
        centrality_values_in = centrality_values_in.unsqueeze(-1)
        matrices_rescaled = matrices * centrality_values_in
        centrality_rescaled =torch.sum(matrices_rescaled, dim=1)

        
        centrality = ((idx+1)/12) * centrality_rescaled + centrality_prec #+ centrality_values_out 
        
        # da qui non serve se non si fa pruning in quel layer
        
        value, mask = torch.topk(centrality, n_patch, largest = True)
        
        mask, idxs = torch.sort(mask, dim=1)

        sorted_centr = value.gather(1, idxs)
                
        return mask, sorted_centr
    
    def create_matrices(self, attn_layer, min_threshold= 0.):
        att_matrix = torch.max(attn_layer, dim=1)[0] # shape -> (B, N + 1, N + 1)
        
        return att_matrix


    def forward(self, x, n_patch):
        centrality = 0
        idx = 0
        mask_list = []
        B, N, D = x.shape
        for attn, ff in self.layers:
            att_emb, att_matrix = attn(x)

            # eliminiamo le teste 
            att_matrix = self.create_matrices(att_matrix) # shape -> (B, K, K)                

            mask, sorted_centr = self.get_mask(att_matrix, centrality, n_patch[idx],idx) # shape -> (B, K)

            centrality = sorted_centr

            mask_list.append(mask)

            mask = mask.unsqueeze(2) # shape -> (B, K, 1)
            mask_emb = mask.expand(B, mask.shape[1], D) # shape -> (B, K, D)

            x = att_emb.gather(1, mask_emb) + x.gather(1, mask_emb)

            x = ff(x) + x

            idx += 1
            
        return self.norm(x), mask_list

class TRAM(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer_simple(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img, n_patch, return_tokens = False):

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(img.device, dtype=x.dtype)

        x, mask = self.transformer(x, n_patch)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        if return_tokens: 
            return self.linear_head(x), mask
        else: 
            return self.linear_head(x)
    
