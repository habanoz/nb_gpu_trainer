def estimate_mfu(self, model, fwdbwd_per_iter, dt, flops_promised):
        """ estimate model flops utilization (MFU) """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = model.get_num_params()
        cfg = model.config
        T = cfg.block_size
        L = cfg.n_layer
        E = cfg.n_embed
        # L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        # flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_token = 6*N + 12*L*E*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        flops_achieved = flops_per_iter * (1.0/dt) # per second
        mfu = flops_achieved / flops_promised
        return mfu