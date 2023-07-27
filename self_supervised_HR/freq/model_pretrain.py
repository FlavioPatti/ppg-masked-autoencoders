import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed
from util.patch_embed import PatchEmbed_org

class MaskedAutoencoderViT_freq(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, type="time",
                 decoder_embed_dim=32, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 audio_exp=False, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, split_pos=False, pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.in_chans = in_chans
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim, type)
        
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding
      

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.no_shift=no_shift

        self.decoder_mode = decoder_mode
        
        #Not used in this model, only to be compatible with finetune mode
        #first istance of regression
        self.conv1 = nn.Conv1d(in_channels=256,out_channels=128,kernel_size=4, stride=4)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=128)
        
        #second istance of regression"
        self.conv2 = nn.Conv1d(in_channels=128,out_channels=64,kernel_size=4, stride=4)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(num_features=64)
        
        #linear layer for predict HR
        self.pooling = nn.AvgPool1d(16)
        self.out_neuron = nn.Linear(in_features=64, out_features=1)
    
        # Transfomer
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)


        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax=nn.LogSoftmax(dim=-1)

        self.mask_t_prob=mask_t_prob
        self.mask_f_prob=mask_f_prob
        self.mask_2d=mask_2d

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 4, H, W)
        x: (N, L, patch_size**2 *4)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
    
        #print(f"h = {h}, w = {w}, p = {p}")
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans)) 

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch: # overlapped patch
            T=101
            F=12
        else:            
            T=32
            F=8
        #x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(noise_t, dim=1) # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1) 
        ids_keep_t = ids_shuffle_t[:,:len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(noise_f, dim=1) # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1) 
        ids_keep_f = ids_shuffle_f[:,:len_keep_f] #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:,:len_keep_f] = 0
        mask_f = torch.gather(mask_f, dim=1, index=ids_restore_f).unsqueeze(1).repeat(1,T,1) # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:,:len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_t).unsqueeze(1).repeat(1,F,1).permute(0,2,1) # N,T,F
        mask = 1-(1-mask_t)*(1-mask_f) # N, T, F

        # get masked x
        id2res=torch.Tensor(list(range(N*T*F))).reshape(N,T,F).to(x.device)
        id2res = id2res + 999*mask # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:,:len_keep_f*len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        # embed patches
       # print(f" x0 = {x.shape}")
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
       # print(f" x1 = {x.shape}")

        # masking: length -> length * mask_ratio
        if mask_2d:
          x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=0.6, mask_f_prob=0.5)
         # print(f" x2 = {x.shape}")
        else:
          x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # print(f" x2 = {x.shape}")
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        #emb = self.encoder_emb(x)
        #print(f" x3 = {x.shape}")
        return x, mask, ids_restore, None

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        #print(f"x4 = {x.shape}")
        x = self.decoder_embed(x)
        #print(f"x5 = {x.shape}")

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        if self.decoder_mode != 0:
            B,L,D=x.shape
            x = x[:,1:,:]
            if self.use_custom_patch:
                x = x.reshape(B,101,12,D)
                x = torch.cat([x,x[:,-1,:].unsqueeze(1)],dim=1) # hack
                x = x.reshape(B,1224,D)
        if self.decoder_mode > 3: # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        #print(f"x6 = {x.shape}")
        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B,102,12,256)
                pred = pred[:,:101,:,:]
                pred = pred.reshape(B,1212,256)
            else:
                pred = pred
        else:
            pred = pred[:,1:, :]
            #print(f"pred = {pred.shape}")

        return pred, None, None #emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 4, H, W]
        pred: [N, L, p*p*4]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
       # print(f"target = {target.shape}")

        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        """ MSE loss """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss, target     

    def forward(self, imgs, mask_ratio=0.1):
        #print(f"imgs = {imgs.shape}")
        emb_enc, mask, ids_restore, _ = self.forward_encoder(imgs, mask_ratio, mask_2d=self.mask_2d)
        #print(f"emb_enc = {emb_enc.shape}")
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore) 
        loss_recon, target = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        return loss_recon, pred, target, emb_enc