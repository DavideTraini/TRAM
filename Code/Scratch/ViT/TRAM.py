import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def delete_row_columns(matrici, indici):
    # Ottieni le dimensioni della matrice e dei batch
    B, L, N, _ = matrici.shape
    K = indici.shape[1]

    # Espandi gli indici per consentire l'indicizzazione batch
    indici_expanded = indici.unsqueeze(1).unsqueeze(3).expand(B, L, K, N)

    # Seleziona le righe corrispondenti
    righe_selezionate = torch.gather(matrici, 2, indici_expanded)

    # Riduci la dimensione delle matrici selezionate
    indici_expanded = indici.unsqueeze(1).unsqueeze(2).expand(B, L, K, K)

    # Seleziona solo le colonne corrispondenti dalle righe selezionate
    nuove_matrici = torch.gather(righe_selezionate, 3, indici_expanded)

    return nuove_matrici


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention_pruning(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer_pruning(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_pruning(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
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
        attns_tokens = att_matrix[:, 1:, 1:] # shape -> (B, N, N)
        
        return attns_tokens


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
            
            # sommiamo 1 a tutti i token e aggiungiamo il CLS (il cls è il token 0)
            mask = mask + 1 # shape -> (B, K)
            # colonna di zeri da concatenare a mask
            zeros_column = torch.zeros(B, 1, dtype=mask.dtype, device=mask.device) # shape -> (B, 1)
            # concatenazione
            mask = torch.cat((zeros_column, mask), dim=1) # shape -> (B, K+1)

            mask = mask.unsqueeze(2) # shape -> (B, K+1, 1)
            mask_emb = mask.expand(B, mask.shape[1], D) # shape -> (B, K+1, D)

            x = att_emb.gather(1, mask_emb) + x.gather(1, mask_emb)

            x = ff(x) + x

            idx += 1
            
        return self.norm(x), mask_list


class TRAM(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_pruning(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, n_patch, return_tokens = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, mask = self.transformer(x, n_patch)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        if return_tokens: return self.mlp_head(x), mask
        else: return self.mlp_head(x)
